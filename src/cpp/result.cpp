#include "result.hpp"

#include <datetime.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "arrow_abi.h"
#include "chunk.hpp"
#include "ducky.hpp"
#include "function.hpp"

using namespace nb::literals;

namespace {

// Cached Python conversion types.
struct PyTypes {
    nb::type_object datetime;  // datetime.datetime (tz-aware branch only)
    nb::object timezone_utc;   // datetime.timezone.utc
    nb::type_object decimal;   // decimal.Decimal
    nb::object decimal_ctx;    // decimal.Context(prec=40)
    nb::type_object uuid;      // uuid.UUID
};

const PyTypes& py_types() {
    static std::atomic<PyTypes*> cached{nullptr};
    static nb::ft_mutex mu;
    return cached_singleton(cached, mu, [] {
        if (!PyDateTimeAPI) PyDateTime_IMPORT;
        PyTypes t;
        nb::module_ datetime = nb::module_::import_("datetime");
        t.datetime = datetime.attr("datetime");
        t.timezone_utc = datetime.attr("timezone").attr("utc");
        nb::module_ decimal = nb::module_::import_("decimal");
        t.decimal = decimal.attr("Decimal");
        t.decimal_ctx = decimal.attr("Context")("prec"_a = 40);
        t.uuid = nb::module_::import_("uuid").attr("UUID");
        return t;
    });
}

nb::object steal_checked(PyObject* obj) {
    if (!obj) throw nb::python_error();
    return nb::steal(obj);
}

// Build a Python int from a 128-bit value: unsigned low half + already-boxed
// (signed or unsigned) high half.
nb::object combine_128(nb::object high, uint64_t low) {
    nb::object shift = steal_checked(PyLong_FromLong(64));
    nb::object hi = steal_checked(PyNumber_Lshift(high.ptr(), shift.ptr()));
    nb::object lo = steal_checked(PyLong_FromUnsignedLongLong(low));
    return steal_checked(PyNumber_Add(hi.ptr(), lo.ptr()));
}

nb::object hugeint_to_py(duckdb_hugeint value) {
    return combine_128(steal_checked(PyLong_FromLongLong(value.upper)), value.lower);
}

nb::object uhugeint_to_py(duckdb_uhugeint value) {
    return combine_128(steal_checked(PyLong_FromUnsignedLongLong(value.upper)), value.lower);
}

// Surface temporal constructor errors from DuckDB values outside Python's range.
nb::object date_to_py(int32_t days) {
    duckdb_date_struct s = duckdb_from_date(duckdb_date{days});
    return steal_checked(PyDate_FromDate(s.year, s.month, s.day));
}

nb::object time_to_py(int64_t micros) {
    duckdb_time_struct s = duckdb_from_time(duckdb_time{micros});
    return steal_checked(PyTime_FromTime(s.hour, s.min, s.sec, s.micros));
}

nb::object timestamp_to_py(int64_t micros, bool utc) {
    duckdb_timestamp_struct s = duckdb_from_timestamp(duckdb_timestamp{micros});
    if (utc) {
        // No CAPI overload takes tzinfo; route through the Python constructor.
        return py_types().datetime(s.date.year, s.date.month, s.date.day, s.time.hour, s.time.min,
                                   s.time.sec, s.time.micros, py_types().timezone_utc);
    }
    return steal_checked(PyDateTime_FromDateAndTime(
        s.date.year, s.date.month, s.date.day, s.time.hour, s.time.min, s.time.sec, s.time.micros));
}

// DuckDB stores UUIDs as int128 with the high bit flipped (so signed ordering
// matches UUID ordering); flip it back for the unsigned 128-bit value.
nb::object uuid_to_py(duckdb_hugeint value) {
    uint64_t high = (uint64_t)value.upper ^ (uint64_t(1) << 63);
    nb::object as_int = combine_128(steal_checked(PyLong_FromUnsignedLongLong(high)), value.lower);
    return py_types().uuid("int"_a = as_int);
}

nb::object decimal_to_py(void* data, idx_t row, duckdb_type internal, uint8_t scale) {
    nb::object unscaled;
    switch (internal) {
        case DUCKDB_TYPE_SMALLINT:
            unscaled = nb::int_(((int16_t*)data)[row]);
            break;
        case DUCKDB_TYPE_INTEGER:
            unscaled = nb::int_(((int32_t*)data)[row]);
            break;
        case DUCKDB_TYPE_BIGINT:
            unscaled = nb::int_(((int64_t*)data)[row]);
            break;
        case DUCKDB_TYPE_HUGEINT:
            unscaled = hugeint_to_py(((duckdb_hugeint*)data)[row]);
            break;
        default:
            throw DuckyError("ducky: unexpected DECIMAL storage type");
    }
    // scaleb shifts the decimal point exactly; the wide context avoids rounding.
    return py_types().decimal(unscaled).attr("scaleb")(-(int)scale, py_types().decimal_ctx);
}

nb::object interval_to_py(duckdb_interval value) {
    // timedelta approximates one month as 30 days; Arrow preserves components.
    int64_t seconds = value.micros / 1000000;
    int64_t micros = value.micros - seconds * 1000000;
    int64_t days = (int64_t)value.days + (int64_t)value.months * 30 + seconds / 86400;
    seconds %= 86400;
    if (days < -999999999 || days > 999999999) {
        throw DuckyError("ducky: INTERVAL value overflows datetime.timedelta");
    }
    return steal_checked(PyDelta_FromDSU((int)days, (int)seconds, (int)micros));
}

idx_t enum_index(void* data, idx_t row, duckdb_type internal) {
    switch (internal) {
        case DUCKDB_TYPE_UTINYINT:
            return ((uint8_t*)data)[row];
        case DUCKDB_TYPE_USMALLINT:
            return ((uint16_t*)data)[row];
        case DUCKDB_TYPE_UINTEGER:
            return ((uint32_t*)data)[row];
        default:
            return 0;
    }
}

nb::object convert_value(duckdb_vector vector, duckdb_logical_type logical, idx_t row);

// Convert one cell using pre-resolved vector metadata.
nb::object convert_cell(duckdb_type tid, duckdb_vector vector, duckdb_logical_type logical,
                        void* data, uint64_t* validity, idx_t row) {
    if (validity && !duckdb_validity_row_is_valid(validity, row)) return nb::none();

    switch (tid) {
        case DUCKDB_TYPE_BOOLEAN:
            return nb::cast(((bool*)data)[row]);
        case DUCKDB_TYPE_TINYINT:
            return nb::cast(((int8_t*)data)[row]);
        case DUCKDB_TYPE_SMALLINT:
            return nb::cast(((int16_t*)data)[row]);
        case DUCKDB_TYPE_INTEGER:
            return nb::cast(((int32_t*)data)[row]);
        case DUCKDB_TYPE_BIGINT:
            return nb::cast(((int64_t*)data)[row]);
        case DUCKDB_TYPE_UTINYINT:
            return nb::cast(((uint8_t*)data)[row]);
        case DUCKDB_TYPE_USMALLINT:
            return nb::cast(((uint16_t*)data)[row]);
        case DUCKDB_TYPE_UINTEGER:
            return nb::cast(((uint32_t*)data)[row]);
        case DUCKDB_TYPE_UBIGINT:
            return nb::cast(((uint64_t*)data)[row]);
        case DUCKDB_TYPE_HUGEINT:
            return hugeint_to_py(((duckdb_hugeint*)data)[row]);
        case DUCKDB_TYPE_UHUGEINT:
            return uhugeint_to_py(((duckdb_uhugeint*)data)[row]);
        case DUCKDB_TYPE_FLOAT:
            return nb::cast(((float*)data)[row]);
        case DUCKDB_TYPE_DOUBLE:
            return nb::cast(((double*)data)[row]);
        case DUCKDB_TYPE_DECIMAL:
            return decimal_to_py(data, row, duckdb_decimal_internal_type(logical),
                                 duckdb_decimal_scale(logical));
        case DUCKDB_TYPE_VARCHAR: {
            auto* s = (duckdb_string_t*)data;
            return nb::str(duckdb_string_t_data(&s[row]), duckdb_string_t_length(s[row]));
        }
        case DUCKDB_TYPE_BLOB: {
            auto* s = (duckdb_string_t*)data;
            return nb::bytes(duckdb_string_t_data(&s[row]), duckdb_string_t_length(s[row]));
        }
        case DUCKDB_TYPE_DATE:
            return date_to_py(((int32_t*)data)[row]);
        case DUCKDB_TYPE_TIME:
            return time_to_py(((int64_t*)data)[row]);
        case DUCKDB_TYPE_TIMESTAMP:
            return timestamp_to_py(((int64_t*)data)[row], false);
        case DUCKDB_TYPE_TIMESTAMP_TZ:
            return timestamp_to_py(((int64_t*)data)[row], true);
        case DUCKDB_TYPE_TIMESTAMP_S:
            return timestamp_to_py(((int64_t*)data)[row] * 1000000, false);
        case DUCKDB_TYPE_TIMESTAMP_MS:
            return timestamp_to_py(((int64_t*)data)[row] * 1000, false);
        case DUCKDB_TYPE_TIMESTAMP_NS:
            return timestamp_to_py(((int64_t*)data)[row] / 1000, false);
        case DUCKDB_TYPE_UUID:
            return uuid_to_py(((duckdb_hugeint*)data)[row]);
        case DUCKDB_TYPE_INTERVAL:
            return interval_to_py(((duckdb_interval*)data)[row]);
        case DUCKDB_TYPE_ENUM: {
            idx_t index = enum_index(data, row, duckdb_enum_internal_type(logical));
            char* value = duckdb_enum_dictionary_value(logical, index);
            nb::str out(value);
            duckdb_free(value);
            return out;
        }
        case DUCKDB_TYPE_LIST: {
            duckdb_list_entry entry = ((duckdb_list_entry*)data)[row];
            duckdb_vector child = duckdb_list_vector_get_child(vector);
            duckdb_logical_type child_type = duckdb_vector_get_column_type(child);
            nb::list items;
            for (idx_t k = 0; k < entry.length; ++k) {
                items.append(convert_value(child, child_type, entry.offset + k));
            }
            duckdb_destroy_logical_type(&child_type);
            return items;
        }
        case DUCKDB_TYPE_ARRAY: {
            idx_t size = duckdb_array_type_array_size(logical);
            duckdb_vector child = duckdb_array_vector_get_child(vector);
            duckdb_logical_type child_type = duckdb_vector_get_column_type(child);
            nb::list items;
            for (idx_t k = 0; k < size; ++k) {
                items.append(convert_value(child, child_type, row * size + k));
            }
            duckdb_destroy_logical_type(&child_type);
            return items;
        }
        case DUCKDB_TYPE_STRUCT: {
            idx_t n = duckdb_struct_type_child_count(logical);
            nb::dict fields;
            for (idx_t c = 0; c < n; ++c) {
                char* name = duckdb_struct_type_child_name(logical, c);
                duckdb_vector child = duckdb_struct_vector_get_child(vector, c);
                duckdb_logical_type child_type = duckdb_vector_get_column_type(child);
                fields[nb::str(name)] = convert_value(child, child_type, row);
                duckdb_destroy_logical_type(&child_type);
                duckdb_free(name);
            }
            return fields;
        }
        case DUCKDB_TYPE_MAP: {
            // Physical layout: LIST(STRUCT(key, value)).
            duckdb_list_entry entry = ((duckdb_list_entry*)data)[row];
            duckdb_vector pairs = duckdb_list_vector_get_child(vector);
            duckdb_vector keys = duckdb_struct_vector_get_child(pairs, 0);
            duckdb_vector values = duckdb_struct_vector_get_child(pairs, 1);
            duckdb_logical_type key_type = duckdb_vector_get_column_type(keys);
            duckdb_logical_type value_type = duckdb_vector_get_column_type(values);
            nb::dict mapping;
            for (idx_t k = 0; k < entry.length; ++k) {
                mapping[convert_value(keys, key_type, entry.offset + k)] =
                    convert_value(values, value_type, entry.offset + k);
            }
            duckdb_destroy_logical_type(&key_type);
            duckdb_destroy_logical_type(&value_type);
            return mapping;
        }
        default:
            throw DuckyError(std::string("ducky: column type ") + duckdb_type_name(tid) +
                             " is not decoded yet");
    }
}

// Resolve vector metadata and convert one nested cell.
nb::object convert_value(duckdb_vector vector, duckdb_logical_type logical, idx_t row) {
    return convert_cell(duckdb_get_type_id(logical), vector, logical,
                        duckdb_vector_get_data(vector), duckdb_vector_get_validity(vector), row);
}

// Return the byte width when DuckDB and Arrow layouts match, otherwise 0.
size_t fast_arrow_width(duckdb_type t) {
    switch (t) {
        case DUCKDB_TYPE_TINYINT:
        case DUCKDB_TYPE_UTINYINT:
            return 1;
        case DUCKDB_TYPE_SMALLINT:
        case DUCKDB_TYPE_USMALLINT:
            return 2;
        case DUCKDB_TYPE_INTEGER:
        case DUCKDB_TYPE_UINTEGER:
        case DUCKDB_TYPE_FLOAT:
        case DUCKDB_TYPE_DATE:  // int32 days since epoch == Arrow date32
            return 4;
        case DUCKDB_TYPE_BIGINT:
        case DUCKDB_TYPE_UBIGINT:
        case DUCKDB_TYPE_DOUBLE:
        case DUCKDB_TYPE_TIME:          // int64 µs == Arrow time64[us]
        case DUCKDB_TYPE_TIMESTAMP:     // int64 µs == Arrow timestamp[us]
        case DUCKDB_TYPE_TIMESTAMP_TZ:  // int64 µs == Arrow timestamp[us, tz=UTC]
        case DUCKDB_TYPE_TIMESTAMP_S:   // int64 s  == Arrow timestamp[s]
        case DUCKDB_TYPE_TIMESTAMP_MS:  // int64 ms == Arrow timestamp[ms]
        case DUCKDB_TYPE_TIMESTAMP_NS:  // int64 ns == Arrow timestamp[ns]
            return 8;
        default:
            return 0;
    }
}

// Arrow stream state for a duckdb_result.
struct ArrowStreamState {
    nb::object owner;              // keeps the Result (and its duckdb_result) alive
    duckdb_result* result;         // borrowed from owner
    duckdb_arrow_options options;  // owned by the stream, destroyed on release
    bool fast;                     // all columns eligible -> one hand-built batch
    bool done;                     // fast path emits exactly one combined batch
    // Exact remaining rows, or 0 when unknown.
    idx_t known_total = 0;
    duckdb_data_chunk scratch = nullptr;  // reused across get_next calls
    std::vector<size_t> widths;           // per-column element width (fast path only)
    std::string last_error;
};

// Make a reusable chunk matching the result schema.
duckdb_data_chunk make_scratch_chunk(duckdb_result* result) {
    idx_t n = duckdb_column_count(result);
    if (n == 0) return nullptr;
    std::vector<duckdb_logical_type> types(n);
    for (idx_t i = 0; i < n; ++i) types[i] = duckdb_column_logical_type(result, i);
    duckdb_data_chunk chunk = duckdb_create_data_chunk(types.data(), n);
    for (idx_t i = 0; i < n; ++i) duckdb_destroy_logical_type(&types[i]);
    return chunk;
}

// Scope owner for a scratch chunk.
struct ScratchChunk {
    duckdb_data_chunk chunk;
    explicit ScratchChunk(duckdb_result* result) : chunk(make_scratch_chunk(result)) {}
    ~ScratchChunk() {
        if (chunk) duckdb_destroy_data_chunk(&chunk);
    }
    ScratchChunk(const ScratchChunk&) = delete;
    ScratchChunk& operator=(const ScratchChunk&) = delete;
};

// Remaining chunks retained when the row count is unknown.
struct ChunkList {
    std::vector<duckdb_data_chunk> chunks;
    std::vector<idx_t> sizes;
    idx_t total = 0;

    explicit ChunkList(duckdb_result& result) {
        while (duckdb_data_chunk ch = duckdb_fetch_chunk(result)) {
            idx_t sz = duckdb_data_chunk_get_size(ch);
            if (sz == 0) {
                duckdb_destroy_data_chunk(&ch);
                continue;
            }
            chunks.push_back(ch);
            sizes.push_back(sz);
            total += sz;
        }
    }
    ~ChunkList() {
        for (duckdb_data_chunk ch : chunks) duckdb_destroy_data_chunk(&ch);
    }
    ChunkList(const ChunkList&) = delete;
    ChunkList& operator=(const ChunkList&) = delete;

    size_t size() const { return chunks.size(); }
    duckdb_vector vector(size_t i, idx_t col) const {
        return duckdb_data_chunk_get_vector(chunks[i], col);
    }
};

// Record + destroy a duckdb_error_data; returns true when an error was present.
bool consume_error(ArrowStreamState* state, duckdb_error_data error) {
    if (!error) return false;
    bool has_error = duckdb_error_data_has_error(error);
    if (has_error) state->last_error = duckdb_error_data_message(error);
    duckdb_destroy_error_data(&error);
    return has_error;
}

// Owns all buffers and child structures for one record batch.
struct FastBatch {
    std::vector<void*> allocations;                      // data + validity buffers to free
    std::vector<ArrowArray> children;                    // one struct per column (stable)
    std::vector<ArrowArray*> child_ptrs;                 // root.children points here
    std::vector<std::array<const void*, 2>> child_bufs;  // [validity, data] per column
    const void* root_buf[1] = {nullptr};                 // struct array: one (null) validity buf

    explicit FastBatch(idx_t ncols) : children(ncols), child_ptrs(ncols), child_bufs(ncols) {}
    ~FastBatch() {
        for (void* p : allocations) std::free(p);
    }
    FastBatch(const FastBatch&) = delete;
    FastBatch& operator=(const FastBatch&) = delete;

    // Allocate and register a buffer owned by this batch.
    void* alloc(size_t bytes) {
        void* p = std::malloc(bytes);
        if (!p) throw std::bad_alloc();
        allocations.push_back(p);
        return p;
    }
};

void fast_release_child(ArrowArray* a) { a->release = nullptr; }  // root owns the memory

void fast_release_root(ArrowArray* a) {
    delete (FastBatch*)a->private_data;  // ~FastBatch frees the buffers
    a->release = nullptr;
    a->private_data = nullptr;
}

// Merge a chunk's NULLs into a lazily allocated Arrow validity bitmap.
void mark_nulls(FastBatch* b, uint8_t*& validity, size_t valid_bytes, uint64_t* val, idx_t sz,
                size_t row_off) {
    if (!validity) {
        validity = (uint8_t*)b->alloc(valid_bytes);
        std::memset(validity, 0xFF, valid_bytes);  // start all-valid
    }
    for (idx_t r = 0; r < sz; ++r) {
        if (!duckdb_validity_row_is_valid(val, r)) {
            size_t bit = row_off + r;
            validity[bit / 8] &= (uint8_t)~(1u << (bit % 8));
        }
    }
}

// Publish the batch as a struct array, transferring ownership to `out`.
void finish_fast_batch(std::unique_ptr<FastBatch> b, ArrowArray* out, idx_t total, idx_t ncols,
                       const std::vector<uint8_t*>& validities) {
    for (idx_t c = 0; c < ncols; ++c) {
        b->child_bufs[c][0] = validities[c];
        ArrowArray& child = b->children[c];
        child = {};
        child.length = (int64_t)total;
        child.null_count = validities[c] ? -1 : 0;  // -1: importer counts lazily
        child.n_buffers = 2;
        child.buffers = b->child_bufs[c].data();
        child.release = fast_release_child;
        b->child_ptrs[c] = &child;
    }

    *out = {};
    out->length = (int64_t)total;
    out->n_buffers = 1;  // struct validity buffer, null = all valid
    out->n_children = (int64_t)ncols;
    out->buffers = b->root_buf;
    out->children = b->child_ptrs.data();
    out->release = fast_release_root;
    out->private_data = b.release();
}

int stream_get_schema(ArrowArrayStream* stream, ArrowSchema* out) {
    auto* state = (ArrowStreamState*)stream->private_data;
    idx_t n = duckdb_column_count(state->result);
    std::vector<duckdb_logical_type> types(n);
    std::vector<const char*> names(n);
    for (idx_t i = 0; i < n; ++i) {
        types[i] = duckdb_column_logical_type(state->result, i);
        names[i] = duckdb_column_name(state->result, i);
    }
    duckdb_error_data error =
        duckdb_to_arrow_schema(state->options, types.data(), names.data(), n, out);
    for (idx_t i = 0; i < n; ++i) duckdb_destroy_logical_type(&types[i]);
    return consume_error(state, error) ? EIO : 0;
}

// Build a combined batch in one pass when the row count is exact.
void build_fast_batch_known_total(ArrowStreamState* state, ArrowArray* out, idx_t total) {
    idx_t ncols = duckdb_column_count(state->result);
    size_t valid_bytes = (total + 7) / 8;

    auto b = std::make_unique<FastBatch>(ncols);
    std::vector<uint8_t*> validities(ncols, nullptr);
    for (idx_t c = 0; c < ncols; ++c) {
        b->child_bufs[c] = {nullptr, b->alloc(total * state->widths[c])};
    }

    ScratchChunk scratch(state->result);
    if (!scratch.chunk) throw std::bad_alloc();
    size_t row_off = 0;
    while (duckdb_fetch_chunk_into(*state->result, scratch.chunk)) {
        idx_t sz = duckdb_data_chunk_get_size(scratch.chunk);
        if (row_off + sz > total) {
            throw std::runtime_error("result produced more rows than duckdb_row_count");
        }
        for (idx_t c = 0; c < ncols; ++c) {
            size_t width = state->widths[c];
            duckdb_vector v = duckdb_data_chunk_get_vector(scratch.chunk, c);
            std::memcpy((char*)b->child_bufs[c][1] + row_off * width, duckdb_vector_get_data(v),
                        sz * width);
            if (uint64_t* val = duckdb_vector_get_validity(v)) {
                mark_nulls(b.get(), validities[c], valid_bytes, val, sz, row_off);
            }
        }
        row_off += sz;
    }
    if (row_off != total) {
        throw std::runtime_error("result produced fewer rows than duckdb_row_count");
    }

    finish_fast_batch(std::move(b), out, total, ncols, validities);
}

// Drain layout-compatible columns into one Arrow batch.
void build_fast_batch(ArrowStreamState* state, ArrowArray* out) {
    if (state->known_total > 0) {
        build_fast_batch_known_total(state, out, state->known_total);
        return;
    }
    idx_t ncols = duckdb_column_count(state->result);

    ChunkList pulled(*state->result);
    if (pulled.total == 0) return;  // no rows -> emit no batch (out->release stays nullptr)
    idx_t total = pulled.total;
    size_t valid_bytes = (total + 7) / 8;

    auto b = std::make_unique<FastBatch>(ncols);
    std::vector<uint8_t*> validities(ncols, nullptr);
    for (idx_t c = 0; c < ncols; ++c) {
        // Concatenate data and validity across chunks.
        size_t width = state->widths[c];
        void* data = b->alloc(total * width);
        b->child_bufs[c] = {nullptr, data};
        size_t row_off = 0;
        for (size_t i = 0; i < pulled.size(); ++i) {
            duckdb_vector v = pulled.vector(i, c);
            std::memcpy((char*)data + row_off * width, duckdb_vector_get_data(v),
                        pulled.sizes[i] * width);
            if (uint64_t* val = duckdb_vector_get_validity(v)) {
                mark_nulls(b.get(), validities[c], valid_bytes, val, pulled.sizes[i], row_off);
            }
            row_off += pulled.sizes[i];
        }
    }

    finish_fast_batch(std::move(b), out, total, ncols, validities);
}

int stream_get_next(ArrowArrayStream* stream, ArrowArray* out) {
    auto* state = (ArrowStreamState*)stream->private_data;
    out->release = nullptr;  // Arrow signals end-of-stream with a released array.
    if (state->fast) {
        if (state->done) return 0;  // the single combined batch was already emitted
        state->done = true;
        try {
            build_fast_batch(state, out);
        } catch (const std::exception& e) {
            state->last_error = e.what();
            return EIO;
        }
        return 0;
    }
    // Convert one reusable DuckDB chunk into Arrow-owned buffers.
    if (!state->scratch) {
        state->scratch = make_scratch_chunk(state->result);
        if (!state->scratch) return 0;
    }
    if (!duckdb_fetch_chunk_into(*state->result, state->scratch)) return 0;  // exhausted
    duckdb_error_data error = duckdb_data_chunk_to_arrow(state->options, state->scratch, out);
    return consume_error(state, error) ? EIO : 0;
}

const char* stream_get_last_error(ArrowArrayStream* stream) {
    auto* state = (ArrowStreamState*)stream->private_data;
    return state->last_error.empty() ? nullptr : state->last_error.c_str();
}

void stream_release(ArrowArrayStream* stream) {
    auto* state = (ArrowStreamState*)stream->private_data;
    if (state) {
        if (state->options) duckdb_destroy_arrow_options(&state->options);
        if (state->scratch) duckdb_destroy_data_chunk(&state->scratch);
        nb::gil_scoped_acquire gil;  // drop the owner reference under the GIL
        delete state;
    }
    stream->private_data = nullptr;
    stream->release = nullptr;
}

void stream_capsule_destructor(PyObject* capsule) {
    auto* stream = (ArrowArrayStream*)PyCapsule_GetPointer(capsule, "arrow_array_stream");
    if (stream) {
        if (stream->release) stream->release(stream);
        free(stream);
    }
}

}  // namespace

nb::object Result::arrow_c_stream(nb::object self, bool allow_fast) {
    if (!handle_ || !handle_->connection) {
        throw DuckyError("ducky: Arrow export is unavailable; the connection is closed");
    }
    // Stream-owned conversion options derived from the result.
    duckdb_arrow_options options = duckdb_result_get_arrow_options(&result_);

    // Combine only bit-compatible layouts when permitted.
    std::vector<size_t> widths(column_count_);
    bool fast = allow_fast && column_count_ > 0;
    for (idx_t c = 0; c < column_count_; ++c) {
        widths[c] = fast_arrow_width(types_[c]);
        fast &= widths[c] != 0;
    }

    auto* stream = (ArrowArrayStream*)calloc(1, sizeof(ArrowArrayStream));
    if (!stream) {
        if (options) duckdb_destroy_arrow_options(&options);
        throw std::bad_alloc();
    }
    auto* state = new ArrowStreamState{};
    state->owner = self;
    state->result = &result_;
    state->options = options;
    state->fast = fast;
    state->done = false;
    // Only untouched materialized results have an exact remaining count.
    state->known_total = (rows_returned_ == 0 && !chunk_ && !duckdb_result_is_streaming(result_))
                             ? duckdb_row_count(&result_)
                             : 0;
    state->widths = std::move(widths);
    stream->get_schema = stream_get_schema;
    stream->get_next = stream_get_next;
    stream->get_last_error = stream_get_last_error;
    stream->release = stream_release;
    stream->private_data = state;

    PyObject* capsule = PyCapsule_New(stream, "arrow_array_stream", stream_capsule_destructor);
    if (!capsule) {
        stream_release(stream);
        free(stream);
        throw nb::python_error();
    }
    return nb::steal(capsule);
}

Result::Result(duckdb_result result, std::shared_ptr<DuckDBHandle> handle)
    : result_(result), handle_(std::move(handle)) {
    // PyDateTimeAPI is static per-TU; prime ours so the *_to_py helpers can
    // use PyDate_FromDate / PyTime_FromTime / etc. unconditionally.
    if (!PyDateTimeAPI) PyDateTime_IMPORT;
    column_count_ = duckdb_column_count(&result_);
    auto schema = std::make_shared<ChunkSchema>();
    schema->names.reserve(column_count_);
    schema->name_to_idx.reserve(column_count_);
    types_.reserve(column_count_);
    column_types_.reserve(column_count_);
    for (idx_t i = 0; i < column_count_; ++i) {
        schema->names.emplace_back(duckdb_column_name(&result_, i));
        schema->name_to_idx.emplace(schema->names.back(), i);
        duckdb_logical_type logical = duckdb_column_logical_type(&result_, i);
        column_types_.push_back(logical);
        types_.push_back(duckdb_get_type_id(logical));
    }
    schema_ = std::move(schema);
}

Result::~Result() {
    release_chunk();
    for (duckdb_logical_type& logical : column_types_) {
        if (logical) duckdb_destroy_logical_type(&logical);
    }
    duckdb_destroy_result(&result_);
}

void Result::release_chunk() {
    if (chunk_) {
        duckdb_destroy_data_chunk(&chunk_);
        chunk_ = nullptr;
    }
    chunk_size_ = 0;
    cursor_ = 0;
}

std::vector<std::string> Result::column_types() const {
    std::vector<std::string> out;
    out.reserve(column_count_);
    for (duckdb_type t : types_) out.emplace_back(duckdb_type_name(t));
    return out;
}

nb::object Result::description() const {
    if (column_count_ == 0) return nb::none();
    nb::list out;
    for (idx_t i = 0; i < column_count_; ++i) {
        // (name, type_code, display_size, internal_size, precision, scale, null_ok)
        out.append(nb::make_tuple(schema_->names[i], duckdb_type_name(types_[i]), nb::none(),
                                  nb::none(), nb::none(), nb::none(), nb::none()));
    }
    return out;
}

bool Result::ensure_row() {
    while (!chunk_ || cursor_ >= chunk_size_) {
        // Refill the cursor's scratch chunk in place.
        if (!chunk_) {
            chunk_ = make_scratch_chunk(&result_);
            if (!chunk_) return false;  // zero-column result
        }
        if (!duckdb_fetch_chunk_into(result_, chunk_)) {
            release_chunk();
            return false;  // no more chunks
        }
        idx_t size = duckdb_data_chunk_get_size(chunk_);
        if (size == 0) continue;
        chunk_size_ = size;
        cursor_ = 0;
        cursors_.resize(column_count_);
        for (idx_t i = 0; i < column_count_; ++i) {
            // Re-resolved per refill: a validity mask can materialize lazily.
            duckdb_vector v = duckdb_data_chunk_get_vector(chunk_, i);
            cursors_[i] = ColumnCursor{v, duckdb_vector_get_data(v), duckdb_vector_get_validity(v)};
        }
    }
    return true;
}

nb::object Result::build_row() {
    idx_t row = cursor_++;
    rows_returned_++;
    // Own the partial tuple across conversion errors.
    nb::object tuple = steal_checked(PyTuple_New((Py_ssize_t)column_count_));
    for (idx_t c = 0; c < column_count_; ++c) {
        const ColumnCursor& cur = cursors_[c];
        nb::object value =
            convert_cell(types_[c], cur.vector, column_types_[c], cur.data, cur.validity, row);
        PyTuple_SET_ITEM(tuple.ptr(), (Py_ssize_t)c, value.release().ptr());
    }
    return tuple;
}

// Binding-level locks serialize cursor mutation.
nb::object Result::fetchone() {
    if (!ensure_row()) return nb::none();
    return build_row();
}

nb::list Result::fetchmany(int64_t size) {
    nb::list out;
    for (int64_t i = 0; i < size && ensure_row(); ++i) out.append(build_row());
    return out;
}

nb::list Result::fetchall() {
    // Pre-size for the known unconsumed materialized rows.
    idx_t total = duckdb_result_is_streaming(result_) ? 0 : duckdb_row_count(&result_);
    if (total > rows_returned_) {
        idx_t remaining = total - rows_returned_;
        PyObject* list = PyList_New((Py_ssize_t)remaining);
        if (!list) throw nb::python_error();
        nb::list out = nb::steal<nb::list>(list);
        for (idx_t i = 0; i < remaining; ++i) {
            if (!ensure_row()) {  // defensive; a materialized count is exact
                if (PyList_SetSlice(list, (Py_ssize_t)i, (Py_ssize_t)remaining, nullptr) < 0) {
                    throw nb::python_error();
                }
                return out;
            }
            PyList_SET_ITEM(list, (Py_ssize_t)i, build_row().release().ptr());
        }
        return out;
    }
    nb::list out;
    while (ensure_row()) out.append(build_row());
    return out;
}

nb::object Result::fetchitem() {
    if (column_count_ != 1) {
        throw DuckyError("ducky: fetchitem() requires a result with exactly 1 column; got " +
                         std::to_string(column_count_));
    }
    if (!ensure_row()) {
        throw DuckyError("ducky: fetchitem() requires exactly 1 row; the result is empty");
    }
    // Decode then confirm no second row remains. value is materialized, so it
    // survives the chunk release that the follow-up ensure_row() may trigger.
    idx_t row = cursor_++;
    rows_returned_++;
    const ColumnCursor& cur = cursors_[0];
    nb::object value =
        convert_cell(types_[0], cur.vector, column_types_[0], cur.data, cur.validity, row);
    if (ensure_row()) {
        throw DuckyError("ducky: fetchitem() requires exactly 1 row; the result has more");
    }
    return value;
}

int64_t Result::rows_changed() {
    // duckdb_rows_changed reinterprets the result as a MaterializedQueryResult
    // internally; guard the streaming case rather than hand it a stream.
    if (duckdb_result_is_streaming(result_)) return 0;
    return (int64_t)duckdb_rows_changed(&result_);
}

duckdb_data_chunk Result::fetch_raw_chunk() {
    duckdb_data_chunk chunk = pull_raw_chunk();
    // Track consumption for later row-count decisions.
    if (chunk) rows_returned_ += duckdb_data_chunk_get_size(chunk);
    return chunk;
}

duckdb_data_chunk Result::pull_raw_chunk() {
    if (duckdb_data_chunk reused = chunk_pool_->take()) {
        // Refill a pooled buffer; false means exhausted.
        if (!duckdb_fetch_chunk_into(result_, reused)) {
            duckdb_destroy_data_chunk(&reused);
            return nullptr;
        }
        return reused;
    }
    return duckdb_fetch_chunk(result_);
}

nb::object Result::fetch_chunk() {
    duckdb_data_chunk chunk = fetch_raw_chunk();
    if (!chunk) return nb::none();
    // The Python wrapper owns the allocated Chunk.
    return nb::cast(new Chunk(chunk, schema_, handle_, chunk_pool_), nb::rv_policy::take_ownership);
}

nb::object Result::fetch_numpy_chunk() {
    duckdb_data_chunk chunk = fetch_raw_chunk();
    if (!chunk) return nb::none();
    // Column views keep their owning Chunk alive.
    nb::object chunk_obj =
        nb::cast(new Chunk(chunk, schema_, handle_, chunk_pool_), nb::rv_policy::take_ownership);
    Chunk& c = nb::cast<Chunk&>(chunk_obj);
    nb::dict out;
    const std::vector<std::string>& names = schema_->names;
    for (idx_t i = 0; i < (idx_t)names.size(); ++i) {
        out[names[i].c_str()] = c.column_at(i, chunk_obj);
    }
    return out;
}

namespace {

// NumPy output layout resolved before scanning.
struct NumpyColumnPlan {
    size_t elem_size;
    nb::dlpack::dtype dtype;
    nb::object struct_view;  // non-null → reinterpret via .view()
};

NumpyColumnPlan plan_column(duckdb_type tid, duckdb_logical_type logical, const std::string& name) {
    if (const TypeSpec* spec = typespec_for(tid)) {
        return NumpyColumnPlan{spec->size, spec->dtype(), nb::object()};
    }
    const StructDtypes& sd = struct_dtypes();
    switch (tid) {
        case DUCKDB_TYPE_HUGEINT:
            return NumpyColumnPlan{sizeof(duckdb_hugeint), nb::dtype<uint8_t>(), sd.hugeint};
        case DUCKDB_TYPE_UHUGEINT:
            return NumpyColumnPlan{sizeof(duckdb_uhugeint), nb::dtype<uint8_t>(), sd.uhugeint};
        case DUCKDB_TYPE_INTERVAL:
            return NumpyColumnPlan{sizeof(duckdb_interval), nb::dtype<uint8_t>(), sd.interval};
        case DUCKDB_TYPE_DECIMAL: {
            duckdb_type internal = duckdb_decimal_internal_type(logical);
            if (internal == DUCKDB_TYPE_HUGEINT) {
                return NumpyColumnPlan{sizeof(duckdb_hugeint), nb::dtype<uint8_t>(), sd.hugeint};
            }
            const TypeSpec* spec = typespec_for(internal);  // always primitive here
            return NumpyColumnPlan{spec->size, spec->dtype(), nb::object()};
        }
        default:
            throw DuckyError(std::string("ducky: to_numpy: column '") + name + "' has type " +
                             duckdb_type_name(tid) +
                             " (no flat ndarray representation; "
                             "use .arrow() for VARCHAR, LIST, STRUCT, MAP, ...)");
    }
}

}  // namespace

nb::object Result::to_numpy(nb::object columns) {
    // Validate selected columns and layouts before scanning.
    std::vector<bool> selected(column_count_, true);
    if (!columns.is_none()) {
        std::fill(selected.begin(), selected.end(), false);
        for (nb::handle item : columns) {
            std::string name = nb::cast<std::string>(item);
            auto it = schema_->name_to_idx.find(name);
            if (it == schema_->name_to_idx.end()) {
                throw DuckyError("ducky: to_numpy: unknown column '" + name + "'");
            }
            selected[it->second] = true;
        }
    }

    std::vector<NumpyColumnPlan> plans(column_count_);
    for (idx_t c = 0; c < column_count_; ++c) {
        if (!selected[c]) continue;
        plans[c] = plan_column(types_[c], column_types_[c], schema_->names[c]);
    }

    // Allocate upfront when the exact remaining row count is known.
    if (rows_returned_ == 0 && !chunk_ && !duckdb_result_is_streaming(result_)) {
        idx_t total = duckdb_row_count(&result_);
        nb::dict out;
        std::vector<char*> bufs(column_count_, nullptr);
        std::vector<nb::object> owners(column_count_);
        for (idx_t c = 0; c < column_count_; ++c) {
            if (!selected[c]) continue;
            size_t bytes = total * plans[c].elem_size;
            void* buf = bytes ? std::malloc(bytes) : nullptr;
            if (bytes && !buf) throw std::bad_alloc();
            // Capsule frees `buf` when the ndarray's owner ref hits zero (or on
            // a throw below, via the owners vector).
            owners[c] = nb::capsule(buf, [](void* p) noexcept { std::free(p); });
            bufs[c] = (char*)buf;
        }

        if (total > 0 && column_count_ > 0) {
            ScratchChunk scratch(&result_);
            if (!scratch.chunk) throw std::bad_alloc();
            idx_t offset = 0;
            while (duckdb_fetch_chunk_into(result_, scratch.chunk)) {
                idx_t sz = duckdb_data_chunk_get_size(scratch.chunk);
                if (offset + sz > total) break;  // caught by the != check below
                for (idx_t c = 0; c < column_count_; ++c) {
                    if (!selected[c]) continue;
                    duckdb_vector v = duckdb_data_chunk_get_vector(scratch.chunk, c);
                    std::memcpy(bufs[c] + offset * plans[c].elem_size, duckdb_vector_get_data(v),
                                sz * plans[c].elem_size);
                }
                offset += sz;
            }
            if (offset != total) {
                throw DuckyError("ducky: to_numpy: result row count changed during the scan");
            }
        }

        for (idx_t c = 0; c < column_count_; ++c) {
            if (!selected[c]) continue;
            const NumpyColumnPlan& p = plans[c];
            nb::object arr;
            if (p.struct_view) {
                size_t byte_shape[1] = {total * p.elem_size};
                nb::ndarray<nb::numpy> raw(bufs[c], 1, byte_shape, owners[c], nullptr, p.dtype);
                arr = nb::cast(raw).attr("view")(p.struct_view);
            } else {
                size_t shape[1] = {(size_t)total};
                arr = nb::cast(
                    nb::ndarray<nb::numpy>(bufs[c], 1, shape, owners[c], nullptr, p.dtype));
            }
            out[schema_->names[c].c_str()] = arr;
        }
        return out;
    }

    // Retain chunks when the total must be discovered first.
    ChunkList pulled(result_);
    idx_t total = pulled.total;

    nb::dict out;
    for (idx_t c = 0; c < column_count_; ++c) {
        if (!selected[c]) continue;
        const NumpyColumnPlan& p = plans[c];
        size_t bytes = total * p.elem_size;
        void* buf = bytes ? std::malloc(bytes) : nullptr;
        if (bytes && !buf) throw std::bad_alloc();
        // Capsule frees `buf` when the ndarray's owner ref hits zero.
        nb::capsule owner(buf, [](void* p) noexcept { std::free(p); });

        size_t offset = 0;
        for (size_t i = 0; i < pulled.size(); ++i) {
            const void* src = duckdb_vector_get_data(pulled.vector(i, c));
            std::memcpy((char*)buf + offset, src, pulled.sizes[i] * p.elem_size);
            offset += pulled.sizes[i] * p.elem_size;
        }

        nb::object arr;
        if (p.struct_view) {
            // Wrap as a flat uint8 buffer (total * elem_size bytes) and let
            // numpy reinterpret it as a structured array of `total` items.
            size_t byte_shape[1] = {bytes};
            nb::ndarray<nb::numpy> raw(buf, 1, byte_shape, owner, nullptr, p.dtype);
            arr = nb::cast(raw).attr("view")(p.struct_view);
        } else {
            size_t shape[1] = {(size_t)total};
            arr = nb::cast(nb::ndarray<nb::numpy>(buf, 1, shape, owner, nullptr, p.dtype));
        }
        out[schema_->names[c].c_str()] = arr;
    }
    return out;
}
