#include "arrow_scan.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "arrow_abi.h"
#include "arrow_util.hpp"
#include "connection.hpp"
#include "ducky.hpp"

namespace nb = nanobind;

namespace {

const char* const SCAN_FUNCTION = "ducky_arrow_scan";
// Column ids at or above this are DuckDB's virtual columns (row id, the
// empty projection of `count(*)`); a scan only has to produce their row count.
constexpr idx_t kVirtualColumnStart = idx_t(1) << 63;

std::string lowercase(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return out;
}

// The registry reference DuckDB holds in its extra-info slots. Heap-allocated so
// the C void*/destroy-callback pair can carry a strong reference.
using RegistryRef = std::shared_ptr<ArrowRegistry>;

void registry_ref_destroy(void* ptr) { delete (RegistryRef*)ptr; }

ArrowRegistry* registry_of(void* extra_info) { return ((RegistryRef*)extra_info)->get(); }

// Plan-time state, immutable after bind. Plans outlive single executions —
// Connection.run()'s statement cache and explicit PreparedStatements re-execute
// the same plan — so nothing consumable (the stream, its cursor) may live here;
// that is per-execution state and goes in ArrowScanState below.
struct ArrowScanBind {
    std::string key;                   // registry key (lowercased name)
    ArrowRegistry* reg = nullptr;      // kept alive by the table function's extra info
    duckdb_connection con = nullptr;   // borrowed
    std::vector<std::string> formats;  // per column: the plan's Arrow layout, checked
    std::vector<std::string> names;    // against each execution's stream
};

void arrow_scan_bind_destroy(void* ptr) { delete (ArrowScanBind*)ptr; }

// One converted batch, shared by the workers emitting windows of it. Output
// vectors reference its buffers, which therefore outlive it as needed.
struct ConvertedBatch {
    duckdb_data_chunk chunk = nullptr;  // nullptr when no real column is projected
    idx_t size = 0;
    ~ConvertedBatch() {
        if (chunk) duckdb_destroy_data_chunk(&chunk);
    }
};

// Per-execution state. Batches are pulled lazily, one at a time, by whichever
// worker needs the next one (holding `mu`, and the GIL while the producer
// runs), so a LIMIT stops reading early and memory stays bounded to the
// batches still being emitted. Workers emit vector-size windows of the current
// batch in parallel without the GIL.
struct ArrowScanState {
    std::mutex mu;
    nb::object capsule;                                 // owns the stream; dropped under the GIL
    ArrowArrayStream* stream = nullptr;                 // borrowed from capsule
    duckdb_connection con = nullptr;                    // borrowed
    duckdb_arrow_converted_schema converted = nullptr;  // projected columns, owned
    // Output column -> projected chunk column, or -1 for a virtual column.
    std::vector<int64_t> output_to_chunk;
    // Source column for each projected chunk column.
    std::vector<idx_t> projection;
    std::shared_ptr<ConvertedBatch> current;
    idx_t next_offset = 0;
    bool done = false;

    // Destroyed via gil_delete (the capsule is a Python object).
    ~ArrowScanState() {
        if (converted) duckdb_destroy_arrow_converted_schema(&converted);
    }
};

[[noreturn]] void throw_stream_error(ArrowArrayStream* stream, const char* what) {
    const char* m = stream->get_last_error ? stream->get_last_error(stream) : nullptr;
    throw DuckyError(std::string("ducky: Arrow stream ") + what + " failed" +
                     (m ? std::string(": ") + m : ""));
}

// A stream capsule for the source registered under `key`: a fresh one for a
// reusable source, the stored one for a one-shot source. With `consume`, a
// one-shot source is marked as read and a second read raises. Requires the GIL.
nb::object source_stream(ArrowRegistry* reg, const std::string& key, bool consume,
                         ArrowArrayStream*& out) {
    nb::object object, capsule;
    {
        std::lock_guard<std::mutex> lock(reg->mu);
        auto it = reg->sources.find(key);
        if (it == reg->sources.end()) {
            throw DuckyError("ducky: no Arrow source registered as '" + key + "'");
        }
        ArrowSource& source = it.value();
        if (source.stream.is_valid()) {
            if (consume && source.consumed) {
                throw DuckyError("ducky: the Arrow source '" + key +
                                 "' is a one-shot reader and was already scanned; register it "
                                 "again, or register a reusable source such as a pyarrow.Table");
            }
            if (consume) source.consumed = true;
            capsule = source.stream;
        } else {
            object = source.object;
        }
    }
    if (!capsule.is_valid()) capsule = object.attr("__arrow_c_stream__")();
    out = (ArrowArrayStream*)PyCapsule_GetPointer(capsule.ptr(), "arrow_array_stream");
    if (!out) {
        PyErr_Clear();
        throw DuckyError("ducky: '" + key + "' did not yield an 'arrow_array_stream' capsule");
    }
    return capsule;
}

// A struct ArrowArray exposing a subset of another's children. Owns the
// original array, which it releases (with every child) when released itself.
struct ProjectedArray {
    ArrowArray original;
    std::vector<ArrowArray*> children;
};

void release_projected(ArrowArray* array) {
    auto* p = (ProjectedArray*)array->private_data;
    if (p->original.release) p->original.release(&p->original);
    delete p;
    array->private_data = nullptr;
    array->release = nullptr;
}

// Move `source` into a view with the given children; `source` is left released.
ArrowArray project_array(ArrowArray& source, const std::vector<idx_t>& columns) {
    auto* p = new ProjectedArray{source, {}};
    source.release = nullptr;
    p->children.reserve(columns.size());
    for (idx_t c : columns) p->children.push_back(p->original.children[c]);
    ArrowArray view = p->original;
    view.n_children = (int64_t)columns.size();
    view.children = p->children.empty() ? nullptr : p->children.data();
    view.dictionary = nullptr;
    view.release = release_projected;
    view.private_data = p;
    return view;
}

// Pull the next non-empty batch, converting its projected columns; nullptr at
// the end. Requires the scan state's lock; takes the GIL for the producer only,
// since the conversion doesn't touch Python.
std::shared_ptr<ConvertedBatch> pull_batch(ArrowScanState& s) {
    for (;;) {
        ArrowArray array{};
        {
            nb::gil_scoped_acquire gil;
            if (s.stream->get_next(s.stream, &array) != 0) {
                throw_stream_error(s.stream, "get_next");
            }
        }
        if (!array.release) return nullptr;  // end of stream
        ArrayRelease array_guard{&array};
        auto batch = std::make_shared<ConvertedBatch>();
        batch->size = (idx_t)array.length;
        if (batch->size == 0) continue;
        if (s.projection.empty()) return batch;  // only virtual columns: rows, no data

        ArrowArray view = project_array(array, s.projection);
        ArrayRelease view_guard{&view};
        std::string err;
        if (drain_arrow_error(
                duckdb_data_chunk_from_arrow(s.con, &view, s.converted, &batch->chunk), err)) {
            throw DuckyError("ducky: failed to convert an Arrow batch to a DuckDB chunk: " + err);
        }
        view.release = nullptr;  // from_arrow took ownership
        return batch;
    }
}

void arrow_scan_bind(duckdb_bind_info info) {
    auto* reg = registry_of(duckdb_bind_get_extra_info(info));
    nb::gil_scoped_acquire gil;
    guard(
        [&] {
            // Sole parameter: the registered table name.
            duckdb_value name_val = duckdb_bind_get_parameter(info, 0);
            char* name_c = duckdb_get_varchar(name_val);
            std::string key = lowercase(name_c);
            duckdb_free(name_c);
            duckdb_destroy_value(&name_val);

            // Only the schema is read here: a one-shot source must keep its
            // batches for the scan, and resolving the column types needs no data.
            ArrowArrayStream* stream = nullptr;
            nb::object capsule = source_stream(reg, key, /*consume=*/false, stream);
            ArrowSchema schema{};
            if (stream->get_schema(stream, &schema) != 0) throw_stream_error(stream, "get_schema");
            SchemaRelease schema_guard{&schema};

            auto* b = new ArrowScanBind{};
            b->key = key;
            b->reg = reg;
            b->con = reg->con;
            duckdb_bind_set_bind_data(info, b, &arrow_scan_bind_destroy);

            std::string err;
            ConvertedSchemaRelease converted;
            if (drain_arrow_error(duckdb_schema_from_arrow(b->con, &schema, &converted.schema),
                                  err)) {
                throw DuckyError("ducky: failed to convert the Arrow schema of '" + key +
                                 "': " + err);
            }
            idx_t n_cols = duckdb_arrow_converted_schema_column_count(converted.schema);
            for (idx_t i = 0; i < n_cols; ++i) {
                ArrowSchema* child = schema.children ? schema.children[i] : nullptr;
                const char* col_name = child && child->name ? child->name : "";
                b->names.emplace_back(col_name);
                b->formats.emplace_back(child && child->format ? child->format : "");
                duckdb_logical_type t =
                    duckdb_arrow_converted_schema_column_type(converted.schema, i);
                duckdb_bind_add_result_column(info, col_name, t);
                duckdb_destroy_logical_type(&t);
            }
        },
        [&](const char* what) { duckdb_bind_set_error(info, what); });
}

void arrow_scan_init(duckdb_init_info info) {
    auto* b = (ArrowScanBind*)duckdb_init_get_bind_data(info);
    nb::gil_scoped_acquire gil;
    guard(
        [&] {
            auto* s = new ArrowScanState{};
            duckdb_init_set_init_data(info, s, &gil_delete<ArrowScanState>);
            s->con = b->con;
            // Fresh stream per execution, looked up again so a re-registered
            // source is picked up.
            s->capsule = source_stream(b->reg, b->key, /*consume=*/true, s->stream);
            ArrowSchema schema{};
            if (s->stream->get_schema(s->stream, &schema) != 0) {
                throw_stream_error(s->stream, "get_schema");
            }
            SchemaRelease schema_guard{&schema};
            // The plan's types came from the schema at bind time.
            bool same = (idx_t)schema.n_children == b->formats.size();
            for (idx_t i = 0; same && i < b->formats.size(); ++i) {
                ArrowSchema* child = schema.children[i];
                same = child && child->format && b->formats[i] == child->format &&
                       b->names[i] == (child->name ? child->name : "");
            }
            if (!same) {
                throw DuckyError("ducky: the Arrow source '" + b->key +
                                 "' changed its schema since the query was planned; prepare "
                                 "the query again");
            }

            // Projection pushdown: convert only the columns the query reads.
            idx_t n_out = duckdb_init_get_column_count(info);
            for (idx_t i = 0; i < n_out; ++i) {
                idx_t col = duckdb_init_get_column_index(info, i);
                if (col >= kVirtualColumnStart || col >= (idx_t)schema.n_children) {
                    s->output_to_chunk.push_back(-1);
                } else {
                    s->output_to_chunk.push_back((int64_t)s->projection.size());
                    s->projection.push_back(col);
                }
            }
            if (!s->projection.empty()) {
                std::vector<ArrowSchema*> children;
                for (idx_t c : s->projection) children.push_back(schema.children[c]);
                ArrowSchema view = schema;  // borrowed children; never released
                view.n_children = (int64_t)children.size();
                view.children = children.data();
                view.release = nullptr;
                std::string err;
                if (drain_arrow_error(duckdb_schema_from_arrow(s->con, &view, &s->converted),
                                      err)) {
                    throw DuckyError("ducky: failed to convert the Arrow schema of '" + b->key +
                                     "': " + err);
                }
            }
            // Workers share the stream; DuckDB caps this at its thread count.
            duckdb_init_set_max_threads(info, 1024);
        },
        [&](const char* what) { duckdb_init_set_error(info, what); });
}

void arrow_scan_function(duckdb_function_info info, duckdb_data_chunk output) {
    auto* s = (ArrowScanState*)duckdb_function_get_init_data(info);

    std::shared_ptr<ConvertedBatch> batch;
    idx_t offset = 0, count = 0;
    bool ok = guard(
        [&] {
            std::lock_guard<std::mutex> lock(s->mu);
            if (!s->current || s->next_offset >= s->current->size) {
                s->current.reset();
                if (!s->done) {
                    s->current = pull_batch(*s);
                    s->next_offset = 0;
                    if (!s->current) s->done = true;
                }
            }
            if (!s->current) return;
            batch = s->current;
            offset = s->next_offset;
            count = std::min(duckdb_vector_size(), batch->size - offset);
            s->next_offset += count;
        },
        [&](const char* what) { duckdb_function_set_error(info, what); });
    if (!ok || !batch) {
        duckdb_data_chunk_set_size(output, 0);
        return;
    }

    // Zero-copy emit: output vectors reference a window of the batch's
    // vectors, whose buffers (for primitive types) are the Arrow source's own.
    // The window keeps a flat vector flat, so consumers that need flat input
    // (an INSERT, for one) don't copy it.
    if (batch->chunk) {
        bool whole = offset == 0 && count == batch->size;
        for (idx_t o = 0; o < (idx_t)s->output_to_chunk.size(); ++o) {
            int64_t c = s->output_to_chunk[o];
            if (c < 0) continue;  // virtual column: only the row count matters
            duckdb_vector out = duckdb_data_chunk_get_vector(output, o);
            duckdb_vector in = duckdb_data_chunk_get_vector(batch->chunk, c);
            if (whole) {
                duckdb_vector_reference_vector(out, in);
            } else {
                duckdb_vector_reference_slice(out, in, offset, count);
            }
        }
    }
    duckdb_data_chunk_set_size(output, count);
}

// When DuckDB can't resolve a table name, rewrite it to ducky_arrow_scan('name')
// if that name is in the registry. Runs during planning; only touches the map
// structure (no nb::object), so no GIL needed here.
void arrow_replacement_scan(duckdb_replacement_scan_info info, const char* table_name, void* data) {
    auto* reg = registry_of(data);
    std::string key = lowercase(table_name);
    {
        std::lock_guard<std::mutex> lock(reg->mu);
        if (reg->sources.find(key) == reg->sources.end()) return;
    }
    duckdb_replacement_scan_set_function_name(info, SCAN_FUNCTION);
    duckdb_value param = duckdb_create_varchar(key.c_str());
    duckdb_replacement_scan_add_parameter(info, param);
    duckdb_destroy_value(&param);
}

// Readers and query results yield their batches once; tables and data frames
// can be streamed again for every scan.
bool is_one_shot(nb::handle obj) {
    return nb::hasattr(obj, "read_next_batch") || nb::hasattr(obj, "fetch_chunk");
}

}  // namespace

ArrowRegistry::~ArrowRegistry() {
    if (sources.empty() || !Py_IsInitialized()) return;
    nb::gil_scoped_acquire gil;
    sources.clear();
}

void register_arrow_source(ArrowRegistry& registry, const std::string& name, nb::object obj) {
    ArrowSource source;
    if (is_one_shot(obj)) {
        source.stream = obj.attr("__arrow_c_stream__")();
    } else {
        source.object = std::move(obj);
    }
    std::lock_guard<std::mutex> lock(registry.mu);
    registry.sources[lowercase(name)] = std::move(source);  // insert / overwrite (GIL held)
}

std::shared_ptr<ArrowRegistry> install_arrow_scan(Connection& con) {
    std::shared_ptr<DuckDBHandle> handle = con.open_handle();
    auto reg = std::make_shared<ArrowRegistry>();
    reg->con = handle->connection;

    duckdb_table_function f = duckdb_create_table_function();
    duckdb_table_function_set_name(f, SCAN_FUNCTION);
    duckdb_logical_type varchar = duckdb_create_logical_type(DUCKDB_TYPE_VARCHAR);
    duckdb_table_function_add_parameter(f, varchar);
    duckdb_destroy_logical_type(&varchar);
    // Strong reference: the function is catalogued on the database and can be
    // re-bound (a cached plan, a PreparedStatement) after the Connection is gone.
    duckdb_table_function_set_extra_info(f, new RegistryRef(reg), &registry_ref_destroy);
    duckdb_table_function_set_bind(f, &arrow_scan_bind);
    duckdb_table_function_set_init(f, &arrow_scan_init);
    duckdb_table_function_set_function(f, &arrow_scan_function);
    duckdb_table_function_supports_projection_pushdown(f, true);

    duckdb_state state;
    {
        // Registration takes the client context lock, which a query running
        // Python callbacks on another thread may hold while it waits for the GIL.
        release_gil_if_held release;
        state = duckdb_register_table_function(handle->connection, f);
        duckdb_destroy_table_function(&f);
    }
    if (state == DuckDBError) {
        throw DuckyError("ducky: failed to register the Arrow scan table function");
    }

    duckdb_add_replacement_scan(handle->database, &arrow_replacement_scan, new RegistryRef(reg),
                                &registry_ref_destroy);
    return reg;
}
