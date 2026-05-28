#include "chunk.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>

#include "ducky.hpp"

namespace {

const char* type_name(duckdb_type type) {
    switch (type) {
        case DUCKDB_TYPE_BOOLEAN:
            return "BOOLEAN";
        case DUCKDB_TYPE_TINYINT:
            return "TINYINT";
        case DUCKDB_TYPE_SMALLINT:
            return "SMALLINT";
        case DUCKDB_TYPE_INTEGER:
            return "INTEGER";
        case DUCKDB_TYPE_BIGINT:
            return "BIGINT";
        case DUCKDB_TYPE_UTINYINT:
            return "UTINYINT";
        case DUCKDB_TYPE_USMALLINT:
            return "USMALLINT";
        case DUCKDB_TYPE_UINTEGER:
            return "UINTEGER";
        case DUCKDB_TYPE_UBIGINT:
            return "UBIGINT";
        case DUCKDB_TYPE_FLOAT:
            return "FLOAT";
        case DUCKDB_TYPE_DOUBLE:
            return "DOUBLE";
        case DUCKDB_TYPE_DATE:
            return "DATE";
        case DUCKDB_TYPE_TIME:
            return "TIME";
        case DUCKDB_TYPE_TIMESTAMP:
            return "TIMESTAMP";
        case DUCKDB_TYPE_TIMESTAMP_S:
            return "TIMESTAMP_S";
        case DUCKDB_TYPE_TIMESTAMP_MS:
            return "TIMESTAMP_MS";
        case DUCKDB_TYPE_TIMESTAMP_NS:
            return "TIMESTAMP_NS";
        case DUCKDB_TYPE_TIMESTAMP_TZ:
            return "TIMESTAMP_TZ";
        case DUCKDB_TYPE_DECIMAL:
            return "DECIMAL";
        case DUCKDB_TYPE_HUGEINT:
            return "HUGEINT";
        case DUCKDB_TYPE_UHUGEINT:
            return "UHUGEINT";
        case DUCKDB_TYPE_VARCHAR:
            return "VARCHAR";
        case DUCKDB_TYPE_BLOB:
            return "BLOB";
        case DUCKDB_TYPE_UUID:
            return "UUID";
        case DUCKDB_TYPE_INTERVAL:
            return "INTERVAL";
        case DUCKDB_TYPE_ENUM:
            return "ENUM";
        case DUCKDB_TYPE_LIST:
            return "LIST";
        case DUCKDB_TYPE_STRUCT:
            return "STRUCT";
        case DUCKDB_TYPE_MAP:
            return "MAP";
        case DUCKDB_TYPE_ARRAY:
            return "ARRAY";
        default:
            return "UNKNOWN";
    }
}

// Maps DuckDB primitive types to a DLPack/numpy dtype matching the chunk's
// in-memory layout. Temporal types are returned as their raw integer storage
// (e.g. DATE -> int32 days since epoch, TIMESTAMP -> int64 microseconds).
// Non-flat types (VARCHAR, BLOB, LIST, STRUCT, MAP, ARRAY, DECIMAL, HUGEINT,
// UUID, INTERVAL) return false; callers should raise.
bool dtype_for(duckdb_type t, nb::dlpack::dtype& out) {
    switch (t) {
        case DUCKDB_TYPE_BOOLEAN:
            out = nb::dtype<bool>();
            return true;
        case DUCKDB_TYPE_TINYINT:
            out = nb::dtype<int8_t>();
            return true;
        case DUCKDB_TYPE_SMALLINT:
            out = nb::dtype<int16_t>();
            return true;
        case DUCKDB_TYPE_INTEGER:
            out = nb::dtype<int32_t>();
            return true;
        case DUCKDB_TYPE_BIGINT:
            out = nb::dtype<int64_t>();
            return true;
        case DUCKDB_TYPE_UTINYINT:
            out = nb::dtype<uint8_t>();
            return true;
        case DUCKDB_TYPE_USMALLINT:
            out = nb::dtype<uint16_t>();
            return true;
        case DUCKDB_TYPE_UINTEGER:
            out = nb::dtype<uint32_t>();
            return true;
        case DUCKDB_TYPE_UBIGINT:
            out = nb::dtype<uint64_t>();
            return true;
        case DUCKDB_TYPE_FLOAT:
            out = nb::dtype<float>();
            return true;
        case DUCKDB_TYPE_DOUBLE:
            out = nb::dtype<double>();
            return true;
        case DUCKDB_TYPE_DATE:
            out = nb::dtype<int32_t>();
            return true;
        case DUCKDB_TYPE_TIME:
        case DUCKDB_TYPE_TIMESTAMP:
        case DUCKDB_TYPE_TIMESTAMP_S:
        case DUCKDB_TYPE_TIMESTAMP_MS:
        case DUCKDB_TYPE_TIMESTAMP_NS:
        case DUCKDB_TYPE_TIMESTAMP_TZ:
            out = nb::dtype<int64_t>();
            return true;
        default:
            return false;
    }
}

}  // namespace

Chunk::Chunk(duckdb_data_chunk chunk, std::vector<std::string> names,
             std::shared_ptr<DuckDBHandle> handle)
    : chunk_(chunk), names_(std::move(names)), handle_(std::move(handle)) {
    size_ = duckdb_data_chunk_get_size(chunk_);
    idx_t n = names_.size();
    types_.reserve(n);
    type_ids_.reserve(n);
    vectors_.reserve(n);
    unpacked_validity_.resize(n);
    for (idx_t i = 0; i < n; ++i) {
        duckdb_vector v = duckdb_data_chunk_get_vector(chunk_, i);
        vectors_.push_back(v);
        duckdb_logical_type logical = duckdb_vector_get_column_type(v);
        types_.push_back(logical);
        type_ids_.push_back(duckdb_get_type_id(logical));
    }
}

Chunk::~Chunk() {
    for (duckdb_logical_type& logical : types_) {
        if (logical) duckdb_destroy_logical_type(&logical);
    }
    if (chunk_) duckdb_destroy_data_chunk(&chunk_);
}

std::vector<std::string> Chunk::column_types() const {
    std::vector<std::string> out;
    out.reserve(type_ids_.size());
    for (duckdb_type t : type_ids_) out.emplace_back(type_name(t));
    return out;
}

idx_t Chunk::resolve(nb::object key) const {
    if (nb::isinstance<nb::str>(key)) {
        std::string name = nb::cast<std::string>(key);
        for (idx_t i = 0; i < names_.size(); ++i) {
            if (names_[i] == name) return i;
        }
        throw DuckyError("ducky: no such column: " + name);
    }
    if (nb::isinstance<nb::int_>(key)) {
        int64_t i = nb::cast<int64_t>(key);
        int64_t n = (int64_t)names_.size();
        if (i < 0) i += n;
        if (i < 0 || i >= n) throw DuckyError("ducky: column index out of range");
        return (idx_t)i;
    }
    throw DuckyError("ducky: column key must be an int or str");
}

nb::object Chunk::column(nb::object key, nb::handle owner) {
    idx_t i = resolve(key);
    nb::dlpack::dtype dtype;
    if (!dtype_for(type_ids_[i], dtype)) {
        throw DuckyError(std::string("ducky: column type ") + type_name(type_ids_[i]) +
                         " has no flat ndarray representation; use .arrow() for this column");
    }
    void* data = duckdb_vector_get_data(vectors_[i]);
    size_t shape[1] = {(size_t)size_};
    // Read-only view: callers shouldn't mutate DuckDB's chunk buffer.
    return nb::cast(nb::ndarray<nb::numpy, nb::ro>(data, 1, shape, owner, nullptr, dtype));
}

nb::object Chunk::validity(nb::object key, nb::handle owner) {
    idx_t i = resolve(key);
    uint64_t* valid = duckdb_vector_get_validity(vectors_[i]);
    if (!valid) return nb::none();
    if (!unpacked_validity_[i]) {
        auto buf = std::make_unique<uint8_t[]>(size_ ? size_ : 1);
        for (idx_t r = 0; r < size_; ++r) {
            buf[r] = duckdb_validity_row_is_valid(valid, r) ? 1 : 0;
        }
        unpacked_validity_[i] = std::move(buf);
    }
    size_t shape[1] = {(size_t)size_};
    return nb::cast(nb::ndarray<nb::numpy, nb::ro>(unpacked_validity_[i].get(), 1, shape, owner,
                                                   nullptr, nb::dtype<uint8_t>()));
}
