#include "arrow_scan.hpp"

#include <nanobind/stl/string.h>

#include <string>

#include "arrow_abi.h"
#include "connection.hpp"
#include "ducky.hpp"

namespace nb = nanobind;

namespace {

const char* const SCAN_FUNCTION = "ducky_arrow_scan";

bool drain_arrow_error(duckdb_error_data err, std::string& out) {
    if (!err) return false;
    bool has = duckdb_error_data_has_error(err);
    if (has) out = duckdb_error_data_message(err);
    duckdb_destroy_error_data(&err);
    return has;
}

// Release-on-scope-exit for Arrow C structs we own.
struct SchemaRelease {
    ArrowSchema* s;
    ~SchemaRelease() {
        if (s && s->release) s->release(s);
    }
};
struct ArrayRelease {
    ArrowArray* a;
    ~ArrayRelease() {
        if (a && a->release) a->release(a);
    }
};

// State carried from bind through the scan; mutated by one thread only since
// the scan sets max_threads = 1 (which also lets us pool the selection vector).
struct ArrowScanBind {
    nb::object capsule;                                 // keeps the Arrow stream alive
    ArrowArrayStream* stream = nullptr;                 // borrowed from `capsule`
    duckdb_arrow_converted_schema converted = nullptr;  // owned
    duckdb_data_chunk pending = nullptr;                // owned; batch currently being emitted
    idx_t cursor = 0;                                   // row offset within `pending`
    bool done = false;                                  // stream exhausted
    duckdb_connection con = nullptr;                    // borrowed
    // Sized to vec_size; reused on every emit to avoid per-chunk alloc/free.
    duckdb_selection_vector sel = nullptr;
};

void arrow_scan_bind_destroy(void* ptr) {
    if (!ptr) return;
    auto* b = (ArrowScanBind*)ptr;
    if (b->pending) duckdb_destroy_data_chunk(&b->pending);
    if (b->converted) duckdb_destroy_arrow_converted_schema(&b->converted);
    if (b->sel) duckdb_destroy_selection_vector(b->sel);
    nb::gil_scoped_acquire gil;  // decref the capsule (its destructor releases the stream)
    delete b;
}

// Pull the next batch into `b->pending` as a converted chunk. Returns false
// at end of stream. Must run with the GIL held — get_next may call into Python.
bool pull_next_batch(ArrowScanBind* b) {
    if (b->pending) {
        duckdb_destroy_data_chunk(&b->pending);
        b->pending = nullptr;
    }
    b->cursor = 0;
    if (b->done) return false;

    ArrowArray array{};
    if (b->stream->get_next(b->stream, &array) != 0) {
        const char* m = b->stream->get_last_error ? b->stream->get_last_error(b->stream) : nullptr;
        throw DuckyError(std::string("ducky: Arrow stream get_next failed") +
                         (m ? std::string(": ") + m : ""));
    }
    if (!array.release) {  // end of stream
        b->done = true;
        return false;
    }
    ArrayRelease array_guard{&array};

    std::string err;
    duckdb_data_chunk chunk = nullptr;
    if (drain_arrow_error(duckdb_data_chunk_from_arrow(b->con, &array, b->converted, &chunk),
                          err)) {
        if (chunk) duckdb_destroy_data_chunk(&chunk);
        throw DuckyError("ducky: failed to convert an Arrow batch to a DuckDB chunk: " + err);
    }
    array.release = nullptr;  // from_arrow took ownership
    b->pending = chunk;
    return true;
}

void arrow_scan_bind(duckdb_bind_info info) {
    auto* reg = (ArrowRegistry*)duckdb_bind_get_extra_info(info);
    nb::gil_scoped_acquire gil;
    guard(
        [&] {
            // Sole parameter: the registered table name.
            duckdb_value name_val = duckdb_bind_get_parameter(info, 0);
            char* name_c = duckdb_get_varchar(name_val);
            std::string name = name_c;
            duckdb_free(name_c);
            duckdb_destroy_value(&name_val);

            nb::object source;
            {
                std::lock_guard<std::mutex> lock(reg->mu);
                auto it = reg->sources.find(name);
                if (it == reg->sources.end()) {
                    throw DuckyError("ducky: no Arrow source registered as '" + name + "'");
                }
                source = it->second;  // incref under the GIL
            }

            nb::object capsule = source.attr("__arrow_c_stream__")();
            auto* stream =
                (ArrowArrayStream*)PyCapsule_GetPointer(capsule.ptr(), "arrow_array_stream");
            if (!stream) {
                PyErr_Clear();
                throw DuckyError("ducky: '" + name +
                                 "' did not yield an 'arrow_array_stream' capsule");
            }

            ArrowSchema schema{};
            if (stream->get_schema(stream, &schema) != 0) {
                const char* m = stream->get_last_error ? stream->get_last_error(stream) : nullptr;
                throw DuckyError(std::string("ducky: Arrow stream get_schema failed") +
                                 (m ? std::string(": ") + m : ""));
            }
            SchemaRelease schema_guard{&schema};

            auto* b = new ArrowScanBind{};
            b->capsule = capsule;
            b->stream = stream;
            b->con = reg->con;
            b->sel = duckdb_create_selection_vector(duckdb_vector_size());
            duckdb_bind_set_bind_data(info, b, &arrow_scan_bind_destroy);

            std::string err;
            if (drain_arrow_error(duckdb_schema_from_arrow(b->con, &schema, &b->converted), err)) {
                throw DuckyError("ducky: failed to convert the Arrow schema of '" + name +
                                 "': " + err);
            }

            // Discover column types by converting the first batch and reading
            // its vector types; names come from the Arrow schema's children.
            if (!pull_next_batch(b)) {
                throw DuckyError("ducky: cannot infer a schema from the empty Arrow source '" +
                                 name + "'");
            }
            idx_t n_cols = duckdb_data_chunk_get_column_count(b->pending);
            for (idx_t i = 0; i < n_cols; ++i) {
                const char* col_name =
                    (schema.children && schema.children[i]) ? schema.children[i]->name : "";
                duckdb_logical_type t =
                    duckdb_vector_get_column_type(duckdb_data_chunk_get_vector(b->pending, i));
                duckdb_bind_add_result_column(info, col_name, t);
                duckdb_destroy_logical_type(&t);
            }
        },
        [&](const char* what) { duckdb_bind_set_error(info, what); });
}

void arrow_scan_init(duckdb_init_info info) {
    // One Arrow stream, one cursor: the scan must run on a single thread.
    duckdb_init_set_max_threads(info, 1);
}

void arrow_scan_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* b = (ArrowScanBind*)duckdb_function_get_bind_data(info);
    nb::gil_scoped_acquire gil;
    guard(
        [&] {
            // Advance to a batch that still has rows, or finish.
            while (!b->pending || b->cursor >= duckdb_data_chunk_get_size(b->pending)) {
                if (!pull_next_batch(b)) {
                    duckdb_data_chunk_set_size(output, 0);
                    return;
                }
            }

            idx_t avail = duckdb_data_chunk_get_size(b->pending) - b->cursor;
            idx_t n = avail < duckdb_vector_size() ? avail : duckdb_vector_size();

            // Emit rows [cursor, cursor + n) via the pooled selection vector;
            // copy-with-selection handles every column type.
            sel_t* sd = duckdb_selection_vector_get_data_ptr(b->sel);
            for (idx_t i = 0; i < n; ++i) sd[i] = (sel_t)(b->cursor + i);

            idx_t n_cols = duckdb_data_chunk_get_column_count(b->pending);
            for (idx_t c = 0; c < n_cols; ++c) {
                duckdb_vector_copy_sel(duckdb_data_chunk_get_vector(b->pending, c),
                                       duckdb_data_chunk_get_vector(output, c), b->sel, n, 0, 0);
            }

            duckdb_data_chunk_set_size(output, n);
            b->cursor += n;
        },
        [&](const char* what) { duckdb_function_set_error(info, what); });
}

// When DuckDB can't resolve a table name, rewrite it to ducky_arrow_scan('name')
// if that name is in the registry. Runs during planning; only touches the map
// structure (no nb::object), so no GIL needed here.
void arrow_replacement_scan(duckdb_replacement_scan_info info, const char* table_name, void* data) {
    auto* reg = (ArrowRegistry*)data;
    {
        std::lock_guard<std::mutex> lock(reg->mu);
        if (reg->sources.find(table_name) == reg->sources.end()) return;
    }
    duckdb_replacement_scan_set_function_name(info, SCAN_FUNCTION);
    duckdb_value param = duckdb_create_varchar(table_name);
    duckdb_replacement_scan_add_parameter(info, param);
    duckdb_destroy_value(&param);
}

}  // namespace

std::unique_ptr<ArrowRegistry> install_arrow_scan(Connection& con) {
    auto reg = std::make_unique<ArrowRegistry>();
    reg->con = con.raw_connection();

    duckdb_table_function f = duckdb_create_table_function();
    duckdb_table_function_set_name(f, SCAN_FUNCTION);
    duckdb_logical_type varchar = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    duckdb_table_function_add_parameter(f, varchar);
    duckdb_destroy_logical_type(&varchar);
    // The registry outlives the function (the Connection owns it), so the
    // extra-info destroy is a no-op (nullptr) — we must not free it here.
    duckdb_table_function_set_extra_info(f, reg.get(), nullptr);
    duckdb_table_function_set_bind(f, &arrow_scan_bind);
    duckdb_table_function_set_init(f, &arrow_scan_init);
    duckdb_table_function_set_function(f, &arrow_scan_function);

    duckdb_state state = duckdb_register_table_function(con.raw_connection(), f);
    duckdb_destroy_table_function(&f);
    if (state == DuckDBError) {
        throw DuckyError("ducky: failed to register the Arrow scan table function");
    }

    duckdb_add_replacement_scan(con.handle()->database, &arrow_replacement_scan, reg.get(),
                                nullptr);
    return reg;
}
