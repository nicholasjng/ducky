#include "chunk.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>

#include "ducky.hpp"
#include "function.hpp"

namespace {

// Maps a DuckDB primitive type to a DLPack/numpy dtype matching the chunk's
// in-memory layout (temporal types come back as their raw integer storage:
// DATE -> int32 days, TIMESTAMP -> int64 micros). Delegates to the shared
// TypeSpec table in function.cpp. Non-flat types (VARCHAR, BLOB, LIST, STRUCT,
// MAP, ARRAY, DECIMAL, HUGEINT, UUID, INTERVAL) return false; callers raise.
bool dtype_for(duckdb_type t, nb::dlpack::dtype& out) {
    if (const TypeSpec* spec = typespec_for(t)) {
        out = spec->dtype();
        return true;
    }
    return false;
}

// Lazily-constructed numpy structured dtypes for HUGEINT, UHUGEINT, and
// INTERVAL. Each is built from Python on first use and intentionally leaked so
// its destructor never runs at interpreter shutdown (same pattern as
// bind_types() in connection.cpp).
struct StructDtypes {
    nb::object hugeint;   // dtype([('lower','<u8'),('upper','<i8')])
    nb::object uhugeint;  // dtype([('lower','<u8'),('upper','<u8')])
    nb::object interval;  // dtype([('months','<i4'),('days','<i4'),('micros','<i8')])
};

const StructDtypes& struct_dtypes() {
    static std::atomic<StructDtypes*> cached{nullptr};
    static nb::ft_mutex mu;
    return cached_singleton(cached, mu, [] {
        nb::module_ np = nb::module_::import_("numpy");
        auto mk = [&](std::initializer_list<std::pair<const char*, const char*>> fields) {
            nb::list spec;
            for (auto& [name, code] : fields) spec.append(nb::make_tuple(name, code));
            return np.attr("dtype")(spec);
        };
        return StructDtypes{
            mk({{"lower", "<u8"}, {"upper", "<i8"}}),
            mk({{"lower", "<u8"}, {"upper", "<u8"}}),
            mk({{"months", "<i4"}, {"days", "<i4"}, {"micros", "<i8"}}),
        };
    });
}

// Wrap `data` (n * itemsize bytes) as a zero-copy uint8 ndarray, then call
// numpy .view(dtype) to reinterpret it as a structured array of n elements.
// `owner` keeps the underlying buffer alive via nanobind's owner mechanism.
nb::object make_struct_view(void* data, size_t n, size_t itemsize, nb::object dtype,
                            nb::handle owner) {
    size_t byte_count = n * itemsize;
    nb::object raw = nb::cast(
        nb::ndarray<nb::numpy, nb::ro>(data, 1, &byte_count, owner, nullptr, nb::dtype<uint8_t>()));
    return raw.attr("view")(dtype);
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
    for (duckdb_type t : type_ids_) out.emplace_back(duckdb_type_name(t));
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
    void* data = duckdb_vector_get_data(vectors_[i]);
    size_t n = (size_t)size_;
    duckdb_type tid = type_ids_[i];

    // Primitive / temporal types — zero-copy ndarray.
    nb::dlpack::dtype dtype;
    if (dtype_for(tid, dtype)) {
        size_t shape[1] = {n};
        return nb::cast(nb::ndarray<nb::numpy, nb::ro>(data, 1, shape, owner, nullptr, dtype));
    }

    // Structured types — zero-copy uint8 buffer reinterpreted via numpy .view().
    const StructDtypes& sd = struct_dtypes();
    if (tid == DUCKDB_TYPE_HUGEINT) {
        return make_struct_view(data, n, sizeof(duckdb_hugeint), sd.hugeint, owner);
    }
    if (tid == DUCKDB_TYPE_UHUGEINT) {
        return make_struct_view(data, n, sizeof(duckdb_uhugeint), sd.uhugeint, owner);
    }
    if (tid == DUCKDB_TYPE_INTERVAL) {
        return make_struct_view(data, n, sizeof(duckdb_interval), sd.interval, owner);
    }
    if (tid == DUCKDB_TYPE_DECIMAL) {
        duckdb_type internal = duckdb_decimal_internal_type(types_[i]);
        if (internal == DUCKDB_TYPE_HUGEINT) {
            return make_struct_view(data, n, sizeof(duckdb_hugeint), sd.hugeint, owner);
        }
        nb::dlpack::dtype idtype;
        (void)dtype_for(internal, idtype);  // internal is always a supported primitive
        size_t shape[1] = {n};
        return nb::cast(nb::ndarray<nb::numpy, nb::ro>(data, 1, shape, owner, nullptr, idtype));
    }

    throw DuckyError(std::string("ducky: column type ") + duckdb_type_name(tid) +
                     " has no flat ndarray representation; use .arrow() for this column");
}

nb::object Chunk::dlpack(nb::object key, nb::handle owner) {
    idx_t i = resolve(key);
    void* data = duckdb_vector_get_data(vectors_[i]);
    size_t n = (size_t)size_;
    duckdb_type tid = type_ids_[i];

    nb::dlpack::dtype dtype;
    if (!dtype_for(tid, dtype)) {
        throw DuckyError(std::string("ducky: column type ") + duckdb_type_name(tid) +
                         " has no flat DLPack representation; use .column() for structured types");
    }
    size_t shape[1] = {n};
    return nb::cast(nb::ndarray<nb::array_api, nb::ro>(data, 1, shape, owner, nullptr, dtype));
}

int Chunk::decimal_scale(nb::object key) {
    idx_t i = resolve(key);
    if (type_ids_[i] != DUCKDB_TYPE_DECIMAL) {
        throw DuckyError(std::string("ducky: decimal_scale() called on non-DECIMAL column '") +
                         names_[i] + "' (" + duckdb_type_name(type_ids_[i]) + ")");
    }
    return (int)duckdb_decimal_scale(types_[i]);
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
