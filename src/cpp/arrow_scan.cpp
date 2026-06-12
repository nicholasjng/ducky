#include "arrow_scan.hpp"

#include <nanobind/stl/string.h>

#include <atomic>
#include <string>
#include <vector>

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

// Plan-time state, immutable after bind. Plans outlive single executions —
// Connection.run()'s statement cache and explicit PreparedStatements re-execute
// the same plan — so nothing consumable (the stream, its cursor) may live here;
// that is per-execution state and goes in ArrowScanState below.
struct ArrowScanBind {
    std::string name;                                   // registered source name
    ArrowRegistry* reg = nullptr;                       // borrowed; the Connection owns it
    duckdb_arrow_converted_schema converted = nullptr;  // owned
    duckdb_connection con = nullptr;                    // borrowed
};

void arrow_scan_bind_destroy(void* ptr) {
    if (!ptr) return;
    auto* b = (ArrowScanBind*)ptr;
    if (b->converted) duckdb_destroy_arrow_converted_schema(&b->converted);
    delete b;
}

// One ≤ vector_size emission window into a converted chunk. Workers claim
// windows through ArrowScanState::next_window, so the windows give the scan
// its parallel granularity.
struct ArrowScanWindow {
    idx_t chunk_idx;
    idx_t offset;
    idx_t count;
};

// Per-execution state: a fresh stream is opened from the registered source in
// arrow_scan_init — so re-executing a (cached or prepared) plan re-streams the
// source — and drained there in full. Draining up front is what makes the scan
// parallel: every batch is converted (a zero-copy wrap of the Arrow buffers
// for primitive types) while the GIL is held once in init, and the scan
// callback then runs GIL-free on any number of worker threads, claiming
// windows off the atomic cursor. The cost is that a streamed source is
// buffered per execution — chunk wraps, not data copies, for the common
// pyarrow.Table case.
struct ArrowScanState {
    std::vector<duckdb_data_chunk> chunks;  // converted batches, owned
    std::vector<ArrowScanWindow> windows;   // emission schedule over `chunks`
    std::atomic<idx_t> next_window{0};
};

void arrow_scan_state_destroy(void* ptr) {
    if (!ptr) return;
    auto* s = (ArrowScanState*)ptr;
    // Chunks co-own the source Arrow arrays; releasing those may call back
    // into pyarrow, so destroy under the GIL.
    nb::gil_scoped_acquire gil;
    for (duckdb_data_chunk& ch : s->chunks) duckdb_destroy_data_chunk(&ch);
    delete s;
}

// Per-thread scratch: the dictionary-slice emit shares the selection vector's
// buffer with the emitted output chunk, so it cannot be shared across workers.
struct ArrowScanLocalState {
    duckdb_selection_vector sel = nullptr;
};

void arrow_scan_local_destroy(void* ptr) {
    if (!ptr) return;
    auto* l = (ArrowScanLocalState*)ptr;
    if (l->sel) duckdb_destroy_selection_vector(l->sel);
    delete l;
}

// Open a stream from the source registered under `name`. Must run with the
// GIL held — __arrow_c_stream__ is a Python call. The stream pointer is
// borrowed from the returned capsule.
nb::object open_stream(ArrowRegistry* reg, const std::string& name, ArrowArrayStream*& out) {
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
    out = (ArrowArrayStream*)PyCapsule_GetPointer(capsule.ptr(), "arrow_array_stream");
    if (!out) {
        PyErr_Clear();
        throw DuckyError("ducky: '" + name + "' did not yield an 'arrow_array_stream' capsule");
    }
    return capsule;
}

// Pull the next batch from `stream` into a converted chunk; nullptr at end of
// stream. Must run with the GIL held — get_next may call into Python.
duckdb_data_chunk convert_next_batch(ArrowArrayStream* stream, duckdb_connection con,
                                     duckdb_arrow_converted_schema converted) {
    ArrowArray array{};
    if (stream->get_next(stream, &array) != 0) {
        const char* m = stream->get_last_error ? stream->get_last_error(stream) : nullptr;
        throw DuckyError(std::string("ducky: Arrow stream get_next failed") +
                         (m ? std::string(": ") + m : ""));
    }
    if (!array.release) return nullptr;  // end of stream
    ArrayRelease array_guard{&array};

    std::string err;
    duckdb_data_chunk chunk = nullptr;
    if (drain_arrow_error(duckdb_data_chunk_from_arrow(con, &array, converted, &chunk), err)) {
        if (chunk) duckdb_destroy_data_chunk(&chunk);
        throw DuckyError("ducky: failed to convert an Arrow batch to a DuckDB chunk: " + err);
    }
    array.release = nullptr;  // from_arrow took ownership
    return chunk;
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

            // Schema discovery uses a throwaway stream, scoped to this bind;
            // the stream the scan consumes is opened per execution in init.
            ArrowArrayStream* stream = nullptr;
            nb::object capsule = open_stream(reg, name, stream);

            ArrowSchema schema{};
            if (stream->get_schema(stream, &schema) != 0) {
                const char* m = stream->get_last_error ? stream->get_last_error(stream) : nullptr;
                throw DuckyError(std::string("ducky: Arrow stream get_schema failed") +
                                 (m ? std::string(": ") + m : ""));
            }
            SchemaRelease schema_guard{&schema};

            auto* b = new ArrowScanBind{};
            b->name = name;
            b->reg = reg;
            b->con = reg->con;
            duckdb_bind_set_bind_data(info, b, &arrow_scan_bind_destroy);

            std::string err;
            if (drain_arrow_error(duckdb_schema_from_arrow(b->con, &schema, &b->converted), err)) {
                throw DuckyError("ducky: failed to convert the Arrow schema of '" + name +
                                 "': " + err);
            }

            // Discover column types by converting the first batch and reading
            // its vector types; names come from the Arrow schema's children.
            duckdb_data_chunk first = convert_next_batch(stream, b->con, b->converted);
            if (!first) {
                throw DuckyError("ducky: cannot infer a schema from the empty Arrow source '" +
                                 name + "'");
            }
            idx_t n_cols = duckdb_data_chunk_get_column_count(first);
            for (idx_t i = 0; i < n_cols; ++i) {
                const char* col_name =
                    (schema.children && schema.children[i]) ? schema.children[i]->name : "";
                duckdb_logical_type t =
                    duckdb_vector_get_column_type(duckdb_data_chunk_get_vector(first, i));
                duckdb_bind_add_result_column(info, col_name, t);
                duckdb_destroy_logical_type(&t);
            }
            duckdb_destroy_data_chunk(&first);
        },
        [&](const char* what) { duckdb_bind_set_error(info, what); });
}

void arrow_scan_init(duckdb_init_info info) {
    auto* b = (ArrowScanBind*)duckdb_init_get_bind_data(info);
    nb::gil_scoped_acquire gil;
    guard(
        [&] {
            // Fresh stream per execution; the source is looked up again so a
            // re-registered source is picked up (its schema must still match
            // the plan's, or batch conversion errors out below). The stream is
            // drained here — see ArrowScanState. The capsule is released at
            // scope exit; the converted chunks co-own the Arrow buffers.
            auto* s = new ArrowScanState{};
            duckdb_init_set_init_data(info, s, &arrow_scan_state_destroy);
            ArrowArrayStream* stream = nullptr;
            nb::object capsule = open_stream(b->reg, b->name, stream);
            idx_t vec_size = duckdb_vector_size();
            for (;;) {
                duckdb_data_chunk ch = convert_next_batch(stream, b->con, b->converted);
                if (!ch) break;
                idx_t size = duckdb_data_chunk_get_size(ch);
                if (size == 0) {
                    duckdb_destroy_data_chunk(&ch);
                    continue;
                }
                idx_t chunk_idx = (idx_t)s->chunks.size();
                s->chunks.push_back(ch);
                for (idx_t off = 0; off < size; off += vec_size) {
                    idx_t count = size - off < vec_size ? size - off : vec_size;
                    s->windows.push_back(ArrowScanWindow{chunk_idx, off, count});
                }
            }
            // One claimable window per worker; DuckDB caps this at its own
            // thread count.
            idx_t windows = (idx_t)s->windows.size();
            duckdb_init_set_max_threads(info, windows ? windows : 1);
        },
        [&](const char* what) { duckdb_init_set_error(info, what); });
}

void arrow_scan_local_init(duckdb_init_info info) {
    // Runs once per worker thread; no Python involved, so no GIL.
    auto* l = new ArrowScanLocalState{};
    duckdb_init_set_init_data(info, l, &arrow_scan_local_destroy);
    l->sel = duckdb_create_selection_vector(duckdb_vector_size());
}

void arrow_scan_function(duckdb_function_info info, duckdb_data_chunk output) {
    // GIL-free and called concurrently from worker threads: all batches were
    // converted in init, so this only claims a window and wires up duckdb
    // vectors. Shared state is read-only except the atomic window cursor.
    auto* s = (ArrowScanState*)duckdb_function_get_init_data(info);
    auto* l = (ArrowScanLocalState*)duckdb_function_get_local_init_data(info);

    idx_t w = s->next_window.fetch_add(1, std::memory_order_relaxed);
    if (w >= (idx_t)s->windows.size()) {
        duckdb_data_chunk_set_size(output, 0);  // this worker is done
        return;
    }
    const ArrowScanWindow& win = s->windows[w];
    duckdb_data_chunk src = s->chunks[win.chunk_idx];

    // Zero-copy emit: each output vector *references* the source chunk's
    // vector — buffer ownership is shared, and for primitive types those
    // buffers are the Arrow source's own (the from_arrow conversion wraps
    // rather than copies them). Windows past row 0 are sliced in as
    // dictionary vectors via this worker's selection vector; rewriting it on
    // the next claim is fine because a source's output is only valid until
    // the worker's next GetData by contract.
    bool aligned = win.offset == 0;
    if (!aligned) {
        sel_t* sd = duckdb_selection_vector_get_data_ptr(l->sel);
        for (idx_t i = 0; i < win.count; ++i) sd[i] = (sel_t)(win.offset + i);
    }
    idx_t n_cols = duckdb_data_chunk_get_column_count(src);
    for (idx_t c = 0; c < n_cols; ++c) {
        duckdb_vector out = duckdb_data_chunk_get_vector(output, c);
        duckdb_vector_reference_vector(out, duckdb_data_chunk_get_vector(src, c));
        if (!aligned) duckdb_slice_vector(out, l->sel, win.count);
    }
    duckdb_data_chunk_set_size(output, win.count);
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
    duckdb_table_function_set_local_init(f, &arrow_scan_local_init);
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
