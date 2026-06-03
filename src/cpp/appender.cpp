#include "appender.hpp"

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstring>
#include <string>

#include "arrow_abi.h"
#include "connection.hpp"
#include "ducky.hpp"
#include "function.hpp"

namespace {

// Returns the flat dtype + element size for numeric / temporal DuckDB types,
// or false for non-flat types (VARCHAR / nested / DECIMAL / HUGEINT / INTERVAL).
// Delegates to the shared TypeSpec table in function.cpp, the same source the
// chunk exporter (chunk.cpp) draws its dtype mapping from.
struct FlatDType {
    nb::dlpack::dtype dtype;
    size_t elem_size;
};

bool flat_dtype_for(duckdb_type t, FlatDType& out) {
    if (const TypeSpec* spec = typespec_for(t)) {
        out.dtype = spec->dtype();
        out.elem_size = spec->size;
        return true;
    }
    return false;
}

// Quote a DuckDB identifier for safe embedding in a SQL string.
std::string quote_ident(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

// Discover column names by running `SELECT * FROM <qualified> LIMIT 0` and
// reading them off the result. We do this once at construction so callers
// can write `append_columns({"name": ndarray})` without an extra DESCRIBE.
std::vector<std::string> discover_column_names(duckdb_connection con,
                                               const std::optional<std::string>& catalog,
                                               const std::optional<std::string>& schema,
                                               const std::string& table) {
    std::string qualified;
    if (catalog) qualified += quote_ident(*catalog) + ".";
    if (schema) qualified += quote_ident(*schema) + ".";
    qualified += quote_ident(table);

    duckdb_result result;
    std::string query = "SELECT * FROM " + qualified + " LIMIT 0";
    if (duckdb_query(con, query.c_str(), &result) == DuckDBError) {
        std::string message = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw DuckyError("ducky: appender column discovery failed: " + message);
    }
    idx_t n = duckdb_column_count(&result);
    std::vector<std::string> names;
    names.reserve(n);
    for (idx_t i = 0; i < n; ++i) {
        names.emplace_back(duckdb_column_name(&result, i));
    }
    duckdb_destroy_result(&result);
    return names;
}

// 1-D, C-contiguous ndarray view of unknown dtype. We do the dtype check
// manually so we can give better error messages than nanobind's templated
// type-mismatch reject.
using AnyArray = nb::ndarray<nb::ro, nb::c_contig>;

bool dtype_eq(nb::dlpack::dtype a, nb::dlpack::dtype b) {
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

}  // namespace

Appender::Appender(Connection& connection, std::string table, std::optional<std::string> schema,
                   std::optional<std::string> catalog) {
    duckdb_connection con = connection.raw_connection();

    duckdb_state state;
    if (catalog) {
        state = duckdb_appender_create_ext(
            con, catalog->c_str(), schema ? schema->c_str() : nullptr, table.c_str(), &handle_);
    } else {
        state = duckdb_appender_create(con, schema ? schema->c_str() : nullptr, table.c_str(),
                                       &handle_);
    }
    if (state == DuckDBError) {
        std::string message = "ducky: failed to create appender for " + quote_ident(table);
        if (handle_) {
            duckdb_error_data err = duckdb_appender_error_data(handle_);
            if (err && duckdb_error_data_has_error(err)) {
                message += ": ";
                message += duckdb_error_data_message(err);
            }
            duckdb_destroy_error_data(&err);
            duckdb_appender_destroy(&handle_);
        }
        throw DuckyError(message);
    }

    // Cache column types up front. Logical types are owned and destroyed in
    // the destructor.
    idx_t n_cols = duckdb_appender_column_count(handle_);
    types_.reserve(n_cols);
    type_ids_.reserve(n_cols);
    for (idx_t i = 0; i < n_cols; ++i) {
        duckdb_logical_type lt = duckdb_appender_column_type(handle_, i);
        type_ids_.push_back(duckdb_get_type_id(lt));
        types_.push_back(lt);  // ownership transferred
    }

    // Names need a separate query — the appender API doesn't expose them.
    try {
        names_ = discover_column_names(con, catalog, schema, table);
    } catch (...) {
        for (auto& lt : types_) duckdb_destroy_logical_type(&lt);
        types_.clear();
        duckdb_appender_destroy(&handle_);
        throw;
    }

    connection_ = connection.handle();
}

Appender::~Appender() {
    close();
    for (auto& lt : types_) {
        if (lt) duckdb_destroy_logical_type(&lt);
    }
    types_.clear();
}

void Appender::ensure_open() const {
    if (!handle_) throw DuckyError("ducky: appender is closed");
}

void Appender::check_state(duckdb_state state, const char* what) {
    if (state != DuckDBError) return;
    std::string message = std::string("ducky: ") + what;
    if (handle_) {
        duckdb_error_data err = duckdb_appender_error_data(handle_);
        if (err && duckdb_error_data_has_error(err)) {
            message += ": ";
            message += duckdb_error_data_message(err);
        }
        duckdb_destroy_error_data(&err);
    }
    throw DuckyError(message);
}

std::vector<std::string> Appender::column_names() const { return names_; }

std::vector<std::string> Appender::column_types() const {
    std::vector<std::string> out;
    out.reserve(type_ids_.size());
    for (auto id : type_ids_) out.emplace_back(duckdb_type_name(id));
    return out;
}

idx_t Appender::resolve(const std::string& name) const {
    for (idx_t i = 0; i < names_.size(); ++i) {
        if (names_[i] == name) return i;
    }
    throw DuckyError("ducky: unknown column '" + name + "' in appender");
}

void Appender::flush() {
    ensure_open();
    check_state(duckdb_appender_flush(handle_), "appender flush failed");
}

void Appender::close() {
    if (!handle_) return;
    // Best-effort: surface the close error, but always destroy.
    duckdb_state state = duckdb_appender_close(handle_);
    std::string error_message;
    if (state == DuckDBError) {
        duckdb_error_data err = duckdb_appender_error_data(handle_);
        if (err && duckdb_error_data_has_error(err)) {
            error_message = duckdb_error_data_message(err);
        }
        duckdb_destroy_error_data(&err);
    }
    duckdb_appender_destroy(&handle_);
    handle_ = nullptr;
    if (state == DuckDBError) {
        throw DuckyError("ducky: appender close failed: " +
                         (error_message.empty() ? std::string("(no message)") : error_message));
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Row API
// ────────────────────────────────────────────────────────────────────────────

namespace {

void append_one(duckdb_appender app, duckdb_type t, nb::handle value, idx_t col) {
    if (value.is_none()) {
        if (duckdb_append_null(app) == DuckDBError) {
            throw DuckyError("ducky: append_null failed at column " + std::to_string(col));
        }
        return;
    }
    duckdb_state state = DuckDBSuccess;
    switch (t) {
        case DUCKDB_TYPE_BOOLEAN:
            state = duckdb_append_bool(app, nb::cast<bool>(value));
            break;
        case DUCKDB_TYPE_TINYINT:
            state = duckdb_append_int8(app, nb::cast<int8_t>(value));
            break;
        case DUCKDB_TYPE_SMALLINT:
            state = duckdb_append_int16(app, nb::cast<int16_t>(value));
            break;
        case DUCKDB_TYPE_INTEGER:
            state = duckdb_append_int32(app, nb::cast<int32_t>(value));
            break;
        case DUCKDB_TYPE_BIGINT:
            state = duckdb_append_int64(app, nb::cast<int64_t>(value));
            break;
        case DUCKDB_TYPE_UTINYINT:
            state = duckdb_append_uint8(app, nb::cast<uint8_t>(value));
            break;
        case DUCKDB_TYPE_USMALLINT:
            state = duckdb_append_uint16(app, nb::cast<uint16_t>(value));
            break;
        case DUCKDB_TYPE_UINTEGER:
            state = duckdb_append_uint32(app, nb::cast<uint32_t>(value));
            break;
        case DUCKDB_TYPE_UBIGINT:
            state = duckdb_append_uint64(app, nb::cast<uint64_t>(value));
            break;
        case DUCKDB_TYPE_FLOAT:
            state = duckdb_append_float(app, nb::cast<float>(value));
            break;
        case DUCKDB_TYPE_DOUBLE:
            state = duckdb_append_double(app, nb::cast<double>(value));
            break;
        case DUCKDB_TYPE_VARCHAR: {
            std::string s = nb::cast<std::string>(value);
            state = duckdb_append_varchar_length(app, s.data(), s.size());
            break;
        }
        case DUCKDB_TYPE_BLOB: {
            nb::bytes b = nb::cast<nb::bytes>(value);
            state = duckdb_append_blob(app, b.c_str(), b.size());
            break;
        }
        default:
            throw DuckyError(std::string("ducky: row-API append for type ") + duckdb_type_name(t) +
                             " not yet supported at column " + std::to_string(col) +
                             "; use append_columns or open a roadmap issue");
    }
    if (state == DuckDBError) {
        throw DuckyError(std::string("ducky: append failed at column ") + std::to_string(col) +
                         " (" + duckdb_type_name(t) + ")");
    }
}

}  // namespace

void Appender::append_row(nb::args values) {
    ensure_open();
    idx_t n_cols = type_ids_.size();
    if (values.size() != n_cols) {
        throw DuckyError("ducky: append_row expected " + std::to_string(n_cols) + " values, got " +
                         std::to_string(values.size()));
    }
    for (idx_t i = 0; i < n_cols; ++i) {
        append_one(handle_, type_ids_[i], values[i], i);
    }
    check_state(duckdb_appender_end_row(handle_), "end_row failed");
}

// ────────────────────────────────────────────────────────────────────────────
// Columnar fast path
// ────────────────────────────────────────────────────────────────────────────

void Appender::append_columns(nb::dict columns, nb::object masks) {
    ensure_open();
    idx_t n_cols = type_ids_.size();
    if (columns.size() == 0) return;

    // Build (col_idx -> ndarray) and (col_idx -> mask ndarray).
    std::vector<AnyArray> col_arrays(n_cols);
    std::vector<bool> col_present(n_cols, false);
    std::vector<AnyArray> mask_arrays(n_cols);
    std::vector<bool> mask_present(n_cols, false);

    int64_t n_rows = -1;

    for (auto [key, val] : columns) {
        std::string name = nb::cast<std::string>(key);
        idx_t idx = resolve(name);
        AnyArray arr = nb::cast<AnyArray>(val);
        if (arr.ndim() != 1) {
            throw DuckyError("ducky: column '" + name + "' must be 1-D, got " +
                             std::to_string(arr.ndim()) + "-D");
        }
        FlatDType expect;
        if (!flat_dtype_for(type_ids_[idx], expect)) {
            throw DuckyError("ducky: columnar append for type " +
                             std::string(duckdb_type_name(type_ids_[idx])) +
                             " not supported in v1 (column '" + name + "'); use append_row");
        }
        if (!dtype_eq(arr.dtype(), expect.dtype)) {
            throw DuckyError("ducky: column '" + name + "' dtype mismatch (target is " +
                             duckdb_type_name(type_ids_[idx]) + ")");
        }
        int64_t len = static_cast<int64_t>(arr.shape(0));
        if (n_rows < 0)
            n_rows = len;
        else if (len != n_rows) {
            throw DuckyError("ducky: column '" + name + "' has length " + std::to_string(len) +
                             ", expected " + std::to_string(n_rows));
        }
        col_arrays[idx] = std::move(arr);
        col_present[idx] = true;
    }

    if (!masks.is_none()) {
        for (auto [key, val] : nb::cast<nb::dict>(masks)) {
            std::string name = nb::cast<std::string>(key);
            idx_t idx = resolve(name);
            AnyArray m = nb::cast<AnyArray>(val);
            if (m.ndim() != 1 || (m.dtype().code != nb::dtype<bool>().code &&
                                  m.dtype().code != nb::dtype<uint8_t>().code)) {
                throw DuckyError("ducky: mask for '" + name + "' must be 1-D bool/uint8");
            }
            if (static_cast<int64_t>(m.shape(0)) != n_rows) {
                throw DuckyError("ducky: mask for '" + name + "' has wrong length");
            }
            mask_arrays[idx] = std::move(m);
            mask_present[idx] = true;
        }
    }

    if (n_rows <= 0) return;

    idx_t vec_size = duckdb_vector_size();

    // Stream chunks of up to vec_size rows.
    for (int64_t offset = 0; offset < n_rows; offset += static_cast<int64_t>(vec_size)) {
        idx_t this_n = static_cast<idx_t>(std::min<int64_t>(vec_size, n_rows - offset));

        duckdb_data_chunk chunk = duckdb_create_data_chunk(types_.data(), n_cols);
        duckdb_data_chunk_set_size(chunk, this_n);

        for (idx_t i = 0; i < n_cols; ++i) {
            duckdb_vector vec = duckdb_data_chunk_get_vector(chunk, i);
            void* dst = duckdb_vector_get_data(vec);

            if (col_present[i]) {
                FlatDType d;
                flat_dtype_for(type_ids_[i], d);
                const uint8_t* src = static_cast<const uint8_t*>(col_arrays[i].data());
                std::memcpy(dst, src + offset * d.elem_size, this_n * d.elem_size);
            } else {
                // Missing column: mark all rows as NULL.
                duckdb_vector_ensure_validity_writable(vec);
                uint64_t* validity = duckdb_vector_get_validity(vec);
                for (idx_t r = 0; r < this_n; ++r) {
                    duckdb_validity_set_row_invalid(validity, r);
                }
            }

            if (mask_present[i]) {
                duckdb_vector_ensure_validity_writable(vec);
                uint64_t* validity = duckdb_vector_get_validity(vec);
                if (mask_arrays[i].dtype().code == nb::dtype<bool>().code) {
                    const bool* m = static_cast<const bool*>(mask_arrays[i].data());
                    for (idx_t r = 0; r < this_n; ++r) {
                        duckdb_validity_set_row_validity(validity, r, m[offset + r]);
                    }
                } else {
                    const uint8_t* m = static_cast<const uint8_t*>(mask_arrays[i].data());
                    for (idx_t r = 0; r < this_n; ++r) {
                        duckdb_validity_set_row_validity(validity, r, m[offset + r] != 0);
                    }
                }
            }
        }

        duckdb_state state = duckdb_append_data_chunk(handle_, chunk);
        duckdb_destroy_data_chunk(&chunk);
        check_state(state, "append_data_chunk failed");
    }
}

namespace {

// Drain a duckdb_error_data into `out`; returns true if it carried an error.
// (Mirrors the helper on the Arrow UDF path in function.cpp.)
bool drain_arrow_error(duckdb_error_data err, std::string& out) {
    if (!err) return false;
    bool has = duckdb_error_data_has_error(err);
    if (has) out = duckdb_error_data_message(err);
    duckdb_destroy_error_data(&err);
    return has;
}

// Release-on-scope-exit for an Arrow C struct we *own* (the stream path fills
// our own stack ArrowSchema/ArrowArray and transfers ownership to us). For the
// __arrow_c_array__ path the PyCapsules own their structs instead, so we don't
// wrap those — see append_arrow.
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
struct ConvertedSchemaRelease {
    duckdb_arrow_converted_schema c = nullptr;
    ~ConvertedSchemaRelease() {
        if (c) duckdb_destroy_arrow_converted_schema(&c);
    }
};

}  // namespace

void Appender::append_arrow(nb::object source) {
    ensure_open();
    duckdb_connection con = connection_->connection;
    if (!con) throw DuckyError("ducky: the appender's connection is closed");

    // Convert one Arrow batch (an ArrowArray matching `converted`) into a data
    // chunk and append it. duckdb_data_chunk_from_arrow takes ownership of the
    // array's buffers, so we null its release afterward to stop the caller's
    // RAII / PyCapsule destructor from double-freeing.
    auto append_batch = [&](duckdb_arrow_converted_schema converted, ArrowArray* array) {
        std::string err;
        duckdb_data_chunk chunk = nullptr;
        if (drain_arrow_error(duckdb_data_chunk_from_arrow(con, array, converted, &chunk), err)) {
            if (chunk) duckdb_destroy_data_chunk(&chunk);
            throw DuckyError("ducky: failed to convert Arrow data to a DuckDB chunk: " + err);
        }
        array->release = nullptr;
        duckdb_state st = duckdb_append_data_chunk(handle_, chunk);
        duckdb_destroy_data_chunk(&chunk);
        check_state(st, "appending an Arrow batch failed");
    };

    if (nb::hasattr(source, "__arrow_c_stream__")) {
        // Stream path: pyarrow Table / RecordBatchReader / polars / pandas-3 /
        // a ducky Result. The capsule owns the stream and releases it when it
        // goes out of scope here; the schema and per-batch arrays it hands us,
        // we own (hence the *Release guards below).
        nb::object capsule = source.attr("__arrow_c_stream__")();
        auto* stream = (ArrowArrayStream*)PyCapsule_GetPointer(capsule.ptr(), "arrow_array_stream");
        if (!stream) {
            PyErr_Clear();
            throw DuckyError(
                "ducky: __arrow_c_stream__ did not return an 'arrow_array_stream' capsule");
        }

        ArrowSchema schema{};
        if (stream->get_schema(stream, &schema) != 0) {
            const char* m = stream->get_last_error ? stream->get_last_error(stream) : nullptr;
            throw DuckyError(std::string("ducky: Arrow stream get_schema failed") +
                             (m ? std::string(": ") + m : ""));
        }
        SchemaRelease schema_guard{&schema};

        std::string err;
        ConvertedSchemaRelease conv;
        if (drain_arrow_error(duckdb_schema_from_arrow(con, &schema, &conv.c), err)) {
            throw DuckyError("ducky: failed to convert the Arrow schema: " + err);
        }

        for (;;) {
            ArrowArray array{};
            if (stream->get_next(stream, &array) != 0) {
                const char* m = stream->get_last_error ? stream->get_last_error(stream) : nullptr;
                throw DuckyError(std::string("ducky: Arrow stream get_next failed") +
                                 (m ? std::string(": ") + m : ""));
            }
            if (!array.release) break;  // end of stream
            ArrayRelease array_guard{&array};
            append_batch(conv.c, &array);
        }
    } else if (nb::hasattr(source, "__arrow_c_array__")) {
        // Single-batch path: a pyarrow RecordBatch. The (schema, array) capsules
        // own their structs and release them when `pair` drops at scope exit, so
        // we don't add our own guards — append_batch only nulls the array's
        // release after consuming it, which the array capsule honors.
        nb::tuple pair = nb::cast<nb::tuple>(source.attr("__arrow_c_array__")());
        auto* schema = (ArrowSchema*)PyCapsule_GetPointer(pair[0].ptr(), "arrow_schema");
        auto* array = (ArrowArray*)PyCapsule_GetPointer(pair[1].ptr(), "arrow_array");
        if (!schema || !array) {
            PyErr_Clear();
            throw DuckyError("ducky: __arrow_c_array__ did not return (schema, array) capsules");
        }
        std::string err;
        ConvertedSchemaRelease conv;
        if (drain_arrow_error(duckdb_schema_from_arrow(con, schema, &conv.c), err)) {
            throw DuckyError("ducky: failed to convert the Arrow schema: " + err);
        }
        append_batch(conv.c, array);
    } else {
        throw DuckyError(
            "ducky: append_arrow() needs an object implementing the Arrow PyCapsule interface "
            "(__arrow_c_stream__ or __arrow_c_array__)");
    }
}
