#include "arrow_scan.hpp"

#include <nanobind/stl/string.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string>
#include <vector>

#include "arrow_abi.h"
#include "arrow_util.hpp"
#include "connection.hpp"
#include "ducky.hpp"

namespace nb = nanobind;

namespace {

const char* const SCAN_FUNCTION = "ducky_arrow_scan";

std::string lowercase(const std::string& s) {
    std::string out = s;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return out;
}

// Heap-allocated so DuckDB's void* and destroy callback can carry a strong
// reference.
using RegistryRef = std::shared_ptr<ArrowRegistry>;

void registry_ref_destroy(void* ptr) { delete (RegistryRef*)ptr; }

duckdb_v2_opaque registry_ref(const std::shared_ptr<ArrowRegistry>& reg) {
    return duckdb_v2_opaque{new RegistryRef(reg), &registry_ref_destroy, nullptr};
}

ArrowRegistry* registry_of(void* user_data) { return ((RegistryRef*)user_data)->get(); }

// Immutable after bind. The statement cache and PreparedStatements re-execute
// a plan, so nothing consumable (the stream, its cursor) may live here. That
// goes in ArrowScanState.
struct ArrowScanBind {
    std::string key;                   // registry key (lowercased name)
    ArrowRegistry* reg = nullptr;      // kept alive by the table function's user data
    std::vector<std::string> formats;  // per column: the plan's Arrow layout, checked
    std::vector<std::string> names;    // against each execution's stream
};

void arrow_scan_bind_destroy(void* ptr) { delete (ArrowScanBind*)ptr; }

// Batches are pulled one at a time by whichever worker needs the next window,
// so a LIMIT stops reading early and memory stays bounded to the batches still
// being emitted. Workers convert vector-size windows of the current batch in
// parallel without the GIL, each with its own importer.
struct ArrowScanState {
    std::mutex mu;
    nb::object capsule;                  // owns the stream, dropped under the GIL
    ArrowArrayStream* stream = nullptr;  // borrowed from capsule
    ArrowSchema schema{};                // the stream's, owned
    // The source column of each output column, and their schema, a view of
    // `schema`.
    std::vector<idx_t> projection;
    std::vector<ArrowSchema*> projected_children;
    ArrowSchema projected{};
    std::shared_ptr<ArrowArray> current;  // projected batch being windowed
    idx_t next_offset = 0;
    bool done = false;

    // Destroyed via gil_delete (the capsule is a Python object).
    ~ArrowScanState() {
        if (schema.release) schema.release(&schema);
    }
};

// Borrows the query's client context, which outlives the scan.
struct ArrowScanLocal {
    duckdb_v2_arrow_importer_handle importer = nullptr;
    ~ArrowScanLocal() { duckdb_v2_arrow_importer_destroy(&importer); }
};

void arrow_scan_local_destroy(void* ptr) { delete (ArrowScanLocal*)ptr; }

[[noreturn]] void throw_stream_error(ArrowArrayStream* stream, const char* what) {
    const char* m = stream->get_last_error ? stream->get_last_error(stream) : nullptr;
    throw DuckyError(std::string("ducky: Arrow stream ") + what + " failed" +
                     (m ? std::string(": ") + m : ""));
}

// With `consume`, a one-shot source is marked as read so a second read raises
// instead of returning nothing. Requires the GIL.
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

// Owns the original array, releasing it with every child when released itself.
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

// A released array at the end. Requires the scan state's lock, and takes the
// GIL only while the producer runs.
ArrowArray next_window(ArrowScanState& s) {
    while (!s.current || s.next_offset >= (idx_t)s.current->length) {
        s.current.reset();
        if (s.done) return ArrowArray{};
        ArrowArray array{};
        {
            nb::gil_scoped_acquire gil;
            if (s.stream->get_next(s.stream, &array) != 0) {
                throw_stream_error(s.stream, "get_next");
            }
        }
        if (!array.release) {  // end of stream
            s.done = true;
            return ArrowArray{};
        }
        ArrayRelease array_guard{&array};
        if (array.length == 0) continue;
        ArrowArray view = project_array(array, s.projection);
        s.current = share_batch(view);
        s.next_offset = 0;
    }
    idx_t count = std::min(duckdb_vector_size(), (idx_t)s.current->length - s.next_offset);
    ArrowArray window = window_of(s.current, s.next_offset, count);
    s.next_offset += count;
    return window;
}

std::string scan_key(duckdb_v2_function_bind_info_handle info) {
    V2Error err;
    duckdb_v2_value_handle value = nullptr;
    err.check(duckdb_v2_function_bind_get_arg_value(info, 0, &value, err.out()));
    const char* text = nullptr;
    idx_t length = 0;
    DUCKDB_V2_ERROR code = duckdb_v2_value_get_varchar(value, &text, &length, err.out());
    std::string key = code == DUCKDB_V2_ERROR_NONE ? lowercase(std::string(text, length)) : "";
    duckdb_v2_value_destroy(&value);
    err.check(code);
    return key;
}

void arrow_scan_bind(duckdb_v2_function_bind_info_handle info,
                     duckdb_v2_table_function_bind_info_handle result,
                     duckdb_v2_context_handle context, duckdb_v2_error_info_handle* err) {
    nb::gil_scoped_acquire gil;
    report(err, [&] {
        V2Error e;
        void* user_data = nullptr;
        e.check(duckdb_v2_function_bind_get_user_data(info, &user_data, e.out()));
        auto bind = std::make_unique<ArrowScanBind>();
        bind->key = scan_key(info);
        bind->reg = registry_of(user_data);

        // Only the schema is read here: a one-shot source must keep its
        // batches for the scan, and resolving the column types needs no data.
        ArrowArrayStream* stream = nullptr;
        nb::object capsule = source_stream(bind->reg, bind->key, /*consume=*/false, stream);
        ArrowSchema schema{};
        if (stream->get_schema(stream, &schema) != 0) throw_stream_error(stream, "get_schema");
        SchemaRelease schema_guard{&schema};

        const std::string failed =
            "ducky: failed to convert the Arrow schema of '" + bind->key + "': ";
        duckdb_v2_arrow_importer_handle importer = nullptr;
        e.check(duckdb_v2_arrow_importer_create(context, &schema, 0, &importer, e.out()), failed);
        duckdb_v2_schema_handle types = nullptr;
        DUCKDB_V2_ERROR code = duckdb_v2_arrow_importer_get_schema(importer, &types, e.out());
        duckdb_v2_arrow_importer_destroy(&importer);
        e.check(code, failed);
        idx_t n_cols = 0;
        code = duckdb_v2_schema_get_count(types, &n_cols, e.out());
        for (idx_t i = 0; code == DUCKDB_V2_ERROR_NONE && i < n_cols; ++i) {
            const char* name = nullptr;
            idx_t length = 0;
            duckdb_v2_logical_type_handle type = nullptr;
            code = duckdb_v2_schema_get_field(types, i, &name, &length, &type, e.out());
            if (code != DUCKDB_V2_ERROR_NONE) break;
            code = duckdb_v2_table_function_bind_add_result_column(result, name, length, type,
                                                                   e.out());
            ArrowSchema* child = schema.children ? schema.children[i] : nullptr;
            bind->names.emplace_back(name, length);
            bind->formats.emplace_back(child && child->format ? child->format : "");
        }
        duckdb_v2_schema_destroy(&types);
        e.check(code, failed);

        duckdb_v2_opaque data{bind.release(), &arrow_scan_bind_destroy, nullptr};
        DUCKDB_V2_ERROR set = duckdb_v2_function_bind_set_bind_data(info, &data, e.out());
        if (set != DUCKDB_V2_ERROR_NONE) arrow_scan_bind_destroy(data.ptr);
        e.check(set);
    });
}

void arrow_scan_init(duckdb_v2_table_function_init_global_info_handle info,
                     duckdb_v2_context_handle context, duckdb_v2_error_info_handle* err) {
    nb::gil_scoped_acquire gil;
    report(err, [&] {
        V2Error e;
        void* bind_data = nullptr;
        e.check(duckdb_v2_table_function_init_global_get_bind_data(info, &bind_data, e.out()));
        auto* b = (ArrowScanBind*)bind_data;
        auto state = std::make_unique<ArrowScanState>();
        ArrowScanState* s = state.get();
        // Fresh stream per execution, looked up again so a re-registered
        // source is picked up.
        s->capsule = source_stream(b->reg, b->key, /*consume=*/true, s->stream);
        ArrowSchema& schema = s->schema;
        if (s->stream->get_schema(s->stream, &schema) != 0) {
            throw_stream_error(s->stream, "get_schema");
        }
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
        idx_t n_out = 0;
        e.check(duckdb_v2_table_function_init_global_get_column_count(info, &n_out, e.out()));
        for (idx_t i = 0; i < n_out; ++i) {
            idx_t col = 0;
            e.check(duckdb_v2_table_function_init_global_get_column_index(info, i, &col, e.out()));
            s->projection.push_back(col);
            s->projected_children.push_back(schema.children[col]);
        }
        s->projected = schema;  // borrowed children, never released
        s->projected.n_children = (int64_t)s->projected_children.size();
        s->projected.children = s->projected_children.data();
        s->projected.release = nullptr;

        // Workers share the stream, and DuckDB caps this at its thread count.
        e.check(duckdb_v2_table_function_init_global_set_max_threads(info, 1024, e.out()));
        duckdb_v2_opaque data{state.release(), &gil_delete<ArrowScanState>, nullptr};
        DUCKDB_V2_ERROR set =
            duckdb_v2_table_function_init_global_set_global_state(info, &data, e.out());
        if (set != DUCKDB_V2_ERROR_NONE) delete s;
        e.check(set);
    });
}

void arrow_scan_init_local(duckdb_v2_table_function_init_local_info_handle info,
                           duckdb_v2_context_handle context, duckdb_v2_error_info_handle* err) {
    report(err, [&] {
        V2Error e;
        void* global_state = nullptr;
        void* bind_data = nullptr;
        e.check(duckdb_v2_table_function_init_local_get_global_state(info, &global_state, e.out()));
        e.check(duckdb_v2_table_function_init_local_get_bind_data(info, &bind_data, e.out()));
        auto* s = (ArrowScanState*)global_state;
        auto local = std::make_unique<ArrowScanLocal>();
        e.check(duckdb_v2_arrow_importer_create(context, &s->projected, duckdb_vector_size(),
                                                &local->importer, e.out()),
                "ducky: failed to convert the Arrow schema of '" +
                    ((ArrowScanBind*)bind_data)->key + "': ");
        duckdb_v2_opaque data{local.release(), &arrow_scan_local_destroy, nullptr};
        DUCKDB_V2_ERROR set =
            duckdb_v2_table_function_init_local_set_local_state(info, &data, e.out());
        if (set != DUCKDB_V2_ERROR_NONE) arrow_scan_local_destroy(data.ptr);
        e.check(set);
    });
}

void arrow_scan_function(duckdb_v2_table_function_exec_info_handle info,
                         duckdb_v2_context_handle context, duckdb_v2_error_info_handle* err) {
    report(err, [&] {
        V2Error e;
        void* global_state = nullptr;
        void* local_state = nullptr;
        e.check(duckdb_v2_table_function_exec_get_global_state(info, &global_state, e.out()));
        e.check(duckdb_v2_table_function_exec_get_local_state(info, &local_state, e.out()));
        auto* s = (ArrowScanState*)global_state;
        auto* local = (ArrowScanLocal*)local_state;
        const std::string failed = "ducky: failed to convert an Arrow batch to a DuckDB chunk: ";

        duckdb_v2_data_chunk_handle chunk = nullptr;
        for (;;) {
            e.check(duckdb_v2_arrow_importer_next_chunk(local->importer, &chunk, e.out()), failed);
            if (chunk) break;
            ArrowArray window;
            {
                std::lock_guard<std::mutex> lock(s->mu);
                window = next_window(*s);
            }
            if (!window.release) return;  // an empty output ends the scan
            ArrayRelease window_guard{&window};
            // Consumed for a zero-copy import, and flushed so that no chunk
            // joins two windows, which would copy.
            e.check(duckdb_v2_arrow_importer_append(local->importer, &window, /*consume=*/true,
                                                    /*flush=*/true, e.out()),
                    failed);
        }

        // Zero-copy emit: the output vectors share the chunk's buffers, whose
        // flat vectors (for primitive types) are views of the Arrow batch's
        // own. Sharing a buffer shares its size, which the engine reads off
        // the first vector.
        duckdb_v2_data_chunk_handle output = nullptr;
        DUCKDB_V2_ERROR code =
            duckdb_v2_table_function_exec_get_output_chunk(info, &output, e.out());
        for (idx_t c = 0; code == DUCKDB_V2_ERROR_NONE && c < s->projection.size(); ++c) {
            duckdb_v2_vector_handle out = nullptr, in = nullptr;
            code = duckdb_v2_data_chunk_get_vector(output, c, &out, e.out());
            if (code == DUCKDB_V2_ERROR_NONE) {
                code = duckdb_v2_data_chunk_get_vector(chunk, c, &in, e.out());
            }
            if (code == DUCKDB_V2_ERROR_NONE) code = duckdb_v2_vector_reference(out, in, e.out());
        }
        duckdb_v2_data_chunk_destroy(&chunk);
        e.check(code);
    });
}

// Runs during planning and touches only the map structure, not Python objects,
// so it needs no GIL.
void arrow_replacement_scan(duckdb_v2_replacement_scan_info_handle info,
                            duckdb_v2_context_handle context, duckdb_v2_error_info_handle* err) {
    report(err, [&] {
        V2Error e;
        void* user_data = nullptr;
        e.check(duckdb_v2_replacement_scan_get_user_data(info, &user_data, e.out()));
        auto* reg = registry_of(user_data);
        // The table name, ignoring any qualification, as v1 handed it over.
        duckdb_v2_qname_handle name = nullptr;
        e.check(duckdb_v2_replacement_scan_get_name(info, &name, e.out()));
        idx_t parts = 0;
        const char* part = nullptr;
        idx_t length = 0;
        DUCKDB_V2_ERROR code = duckdb_v2_qname_get_part_count(name, &parts, e.out());
        if (code == DUCKDB_V2_ERROR_NONE) {
            code = duckdb_v2_qname_get_part(name, parts - 1, &part, &length, e.out());
        }
        std::string key = code == DUCKDB_V2_ERROR_NONE ? lowercase(std::string(part, length)) : "";
        duckdb_v2_qname_destroy(&name);
        e.check(code);
        {
            std::lock_guard<std::mutex> lock(reg->mu);
            if (reg->sources.find(key) == reg->sources.end()) return;
        }

        const char* function = SCAN_FUNCTION;
        idx_t function_length = std::strlen(SCAN_FUNCTION);
        e.check(duckdb_v2_qname_create(&function, &function_length, 1, &name, e.out()));
        code = duckdb_v2_replacement_scan_set_function_name(info, name, e.out());
        duckdb_v2_qname_destroy(&name);
        e.check(code);
        duckdb_v2_factory_handle factory = nullptr;
        e.check(duckdb_v2_context_get_factory(context, &factory, e.out()));
        duckdb_v2_value_handle arg = nullptr;
        e.check(duckdb_v2_value_create_varchar(factory, key.data(), key.size(), &arg, e.out()));
        code = duckdb_v2_replacement_scan_add_argument(info, arg, e.out());
        duckdb_v2_value_destroy(&arg);
        e.check(code);
    });
}

// Readers and query results yield their batches once, while tables and data
// frames can be streamed again for every scan.
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
    duckdb_v2_connection_handle connection = handle->connection_v2;
    auto reg = std::make_shared<ArrowRegistry>();

    const std::string failed = "ducky: failed to register the Arrow scan table function: ";
    V2Error err;
    auto check = [&](DUCKDB_V2_ERROR code) { err.check(code, failed); };
    duckdb_v2_factory_handle factory = nullptr;
    check(duckdb_v2_connection_get_factory(connection, &factory, err.out()));
    // Strong references: the function and the replacement scan are catalogued
    // on the database and can be re-bound (a cached plan, a PreparedStatement)
    // after the Connection is gone. A setter that fails leaves the data with us.
    auto set_registry = [&](auto setter, auto handle) {
        duckdb_v2_opaque data = registry_ref(reg);
        DUCKDB_V2_ERROR code = setter(handle, &data, err.out());
        if (code != DUCKDB_V2_ERROR_NONE) registry_ref_destroy(data.ptr);
        check(code);
    };

    {
        duckdb_v2_table_function_handle f = nullptr;
        check(duckdb_v2_table_function_create(factory, &f, err.out()));
        std::unique_ptr<duckdb_v2_table_function_handle, void (*)(duckdb_v2_table_function_handle*)>
            f_guard(&f, [](duckdb_v2_table_function_handle* p) {
                duckdb_v2_table_function_destroy(p);
            });
        set_registry(&duckdb_v2_table_function_set_user_data, f);
        check(duckdb_v2_table_function_set_name(f, SCAN_FUNCTION, std::strlen(SCAN_FUNCTION),
                                                err.out()));
        duckdb_v2_function_signature_handle sig = nullptr;
        check(duckdb_v2_table_function_get_signature(f, &sig, err.out()));
        duckdb_v2_logical_type_handle varchar = nullptr;
        check(duckdb_v2_factory_create_type_from_id(factory, DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR,
                                                    nullptr, nullptr, nullptr, 0, &varchar,
                                                    err.out()));
        DUCKDB_V2_ERROR code = duckdb_v2_function_signature_add_parameter(
            sig, "name", 4, varchar, nullptr, DUCKDB_V2_FUNCTION_PARAMETER_KIND_STANDARD,
            err.out());
        duckdb_v2_logical_type_destroy(&varchar);
        check(code);
        check(duckdb_v2_table_function_set_bind_callback(f, &arrow_scan_bind, err.out()));
        check(duckdb_v2_table_function_set_init_global_callback(f, &arrow_scan_init, err.out()));
        check(
            duckdb_v2_table_function_set_init_local_callback(f, &arrow_scan_init_local, err.out()));
        check(duckdb_v2_table_function_set_exec_callback(f, &arrow_scan_function, err.out()));
        check(duckdb_v2_table_function_set_projection_pushdown(f, true, err.out()));
        {
            // Registration takes the client context lock, which a query running
            // Python callbacks on another thread may hold while it waits for
            // the GIL.
            release_gil_if_held release;
            code = duckdb_v2_connection_register_table_function(connection, f, err.out());
        }
        check(code);
    }

    duckdb_v2_replacement_scan_handle scan = nullptr;
    check(duckdb_v2_replacement_scan_create(factory, &scan, err.out()));
    std::unique_ptr<duckdb_v2_replacement_scan_handle, void (*)(duckdb_v2_replacement_scan_handle*)>
        scan_guard(&scan, [](duckdb_v2_replacement_scan_handle* p) {
            duckdb_v2_replacement_scan_destroy(p);
        });
    set_registry(&duckdb_v2_replacement_scan_set_user_data, scan);
    check(duckdb_v2_replacement_scan_set_callback(scan, &arrow_replacement_scan, err.out()));
    DUCKDB_V2_ERROR code;
    {
        release_gil_if_held release;
        code = duckdb_v2_connection_register_replacement_scan(connection, scan, err.out());
    }
    check(code);
    return reg;
}
