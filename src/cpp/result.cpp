#include "result.hpp"

#include <datetime.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>

#include "arrow_abi.h"
#include "chunk.hpp"
#include "ducky.hpp"
#include "function.hpp"

using namespace nb::literals;

namespace {

// Cached Python conversion types.
struct PyTypes {
    nb::object timezone_utc;    // datetime.timezone.utc
    nb::type_object timezone;   // datetime.timezone
    nb::type_object decimal;    // decimal.Decimal
    nb::object decimal_ctx;     // decimal.Context(prec=40)
    nb::type_object uuid;       // uuid.UUID
    nb::object int_from_bytes;  // int.from_bytes
    // ±infinity dates and timestamps map to the representable extremes, as in
    // duckdb-python.
    nb::object date_min, date_max;
    nb::object datetime_min, datetime_max;
    nb::object datetime_min_utc, datetime_max_utc;
};

const PyTypes& py_types() {
    static std::atomic<PyTypes*> cached{nullptr};
    static nb::ft_mutex mu;
    return cached_singleton(cached, mu, [] {
        if (!PyDateTimeAPI) PyDateTime_IMPORT;
        PyTypes t;
        nb::module_ datetime = nb::module_::import_("datetime");
        t.timezone = datetime.attr("timezone");
        t.timezone_utc = t.timezone.attr("utc");
        nb::module_ decimal = nb::module_::import_("decimal");
        t.decimal = decimal.attr("Decimal");
        t.decimal_ctx = decimal.attr("Context")("prec"_a = 40);
        t.uuid = nb::module_::import_("uuid").attr("UUID");
        t.int_from_bytes = nb::module_::import_("builtins").attr("int").attr("from_bytes");
        nb::object date = datetime.attr("date");
        nb::object dt = datetime.attr("datetime");
        t.date_min = date.attr("min");
        t.date_max = date.attr("max");
        t.datetime_min = dt.attr("min");
        t.datetime_max = dt.attr("max");
        t.datetime_min_utc = t.datetime_min.attr("replace")("tzinfo"_a = t.timezone_utc);
        t.datetime_max_utc = t.datetime_max.attr("replace")("tzinfo"_a = t.timezone_utc);
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

int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q;
}

// Finite dates outside Python's range raise ValueError; ±infinity maps to
// date.min / date.max.
nb::object date_to_py(int32_t days) {
    if (!duckdb_is_finite_date(duckdb_date{days})) {
        return days > 0 ? py_types().date_max : py_types().date_min;
    }
    duckdb_date_struct s = duckdb_from_date(duckdb_date{days});
    return steal_checked(PyDate_FromDate(s.year, s.month, s.day));
}

nb::object time_to_py(int64_t micros) {
    duckdb_time_struct s = duckdb_from_time(duckdb_time{micros});
    return steal_checked(PyTime_FromTime(s.hour, s.min, s.sec, s.micros));
}

// `timezones` caches one datetime.timezone per UTC offset.
nb::object time_tz_to_py(uint64_t bits, tsl::robin_map<int32_t, nb::object>& timezones) {
    duckdb_time_tz_struct s = duckdb_from_time_tz(duckdb_time_tz{bits});
    auto it = timezones.find(s.offset);
    if (it == timezones.end()) {
        nb::object delta = steal_checked(PyDelta_FromDSU(0, s.offset, 0));
        it = timezones.emplace(s.offset, py_types().timezone(delta)).first;
    }
    const nb::object& tz = it->second;
    return steal_checked(PyDateTimeAPI->Time_FromTime(
        s.time.hour, s.time.min, s.time.sec, s.time.micros, tz.ptr(), PyDateTimeAPI->TimeType));
}

// `micros` must be finite.
nb::object timestamp_micros_to_py(int64_t micros, bool utc) {
    duckdb_timestamp_struct s = duckdb_from_timestamp(duckdb_timestamp{micros});
    PyObject* tz = utc ? py_types().timezone_utc.ptr() : Py_None;
    return steal_checked(PyDateTimeAPI->DateTime_FromDateAndTime(
        s.date.year, s.date.month, s.date.day, s.time.hour, s.time.min, s.time.sec, s.time.micros,
        tz, PyDateTimeAPI->DateTimeType));
}

nb::object infinite_timestamp(int64_t raw, bool utc) {
    const PyTypes& t = py_types();
    if (raw > 0) return utc ? t.datetime_max_utc : t.datetime_max;
    return utc ? t.datetime_min_utc : t.datetime_min;
}

nb::object timestamp_to_py(int64_t micros, bool utc) {
    if (!duckdb_is_finite_timestamp(duckdb_timestamp{micros})) {
        return infinite_timestamp(micros, utc);
    }
    return timestamp_micros_to_py(micros, utc);
}

// TIMESTAMP_S / TIMESTAMP_MS: scale to microseconds without overflowing.
nb::object scaled_timestamp_to_py(int64_t value, int64_t micros_per_unit, bool finite) {
    if (!finite) return infinite_timestamp(value, false);
    if (value > std::numeric_limits<int64_t>::max() / micros_per_unit ||
        value < std::numeric_limits<int64_t>::min() / micros_per_unit) {
        throw nb::value_error("ducky: timestamp is out of the range of datetime.datetime");
    }
    return timestamp_micros_to_py(value * micros_per_unit, false);
}

nb::object timestamp_ns_to_py(int64_t nanos) {
    if (!duckdb_is_finite_timestamp_ns(duckdb_timestamp_ns{nanos})) {
        return infinite_timestamp(nanos, false);
    }
    // datetime has microsecond resolution: round towards negative infinity so
    // instants before the epoch don't move forward.
    return timestamp_micros_to_py(floor_div(nanos, 1000), false);
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
    // Normalize in 64 bits: PyDelta_FromDSU takes ints, so seconds and
    // microseconds must already be in range.
    int64_t seconds = floor_div(value.micros, 1000000);
    int64_t micros = value.micros - seconds * 1000000;
    int64_t days = (int64_t)value.days + (int64_t)value.months * 30 + floor_div(seconds, 86400);
    seconds -= floor_div(seconds, 86400) * 86400;
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

// BIT: one byte of padding-bit count, then the bits MSB first. Rendered as a
// '0'/'1' string, as DuckDB prints it.
nb::object bit_to_py(const duckdb_string_t& value) {
    const uint8_t* bytes = (const uint8_t*)duckdb_string_t_data((duckdb_string_t*)&value);
    idx_t length = duckdb_string_t_length(value);
    if (length < 1) return nb::str("");
    idx_t padding = bytes[0];
    std::string out;
    out.reserve((length - 1) * 8);
    for (idx_t byte = 1; byte < length; ++byte) {
        for (idx_t bit = (byte == 1 ? padding : 0); bit < 8; ++bit) {
            out.push_back((bytes[byte] >> (7 - bit)) & 1 ? '1' : '0');
        }
    }
    return nb::str(out.data(), out.size());
}

// BIGNUM: a 3-byte header whose top bit is set for non-negative values, then
// the big-endian magnitude (bytes inverted for negative values).
nb::object bignum_to_py(const duckdb_string_t& value) {
    const uint8_t* bytes = (const uint8_t*)duckdb_string_t_data((duckdb_string_t*)&value);
    idx_t length = duckdb_string_t_length(value);
    if (length < 4) throw DuckyError("ducky: malformed BIGNUM value");
    bool negative = (bytes[0] & 0x80) == 0;
    std::string magnitude((const char*)bytes + 3, length - 3);
    if (negative) {
        for (char& b : magnitude) b = (char)~(uint8_t)b;
    }
    nb::object result =
        py_types().int_from_bytes(nb::bytes(magnitude.data(), magnitude.size()), "big");
    return negative ? steal_checked(PyNumber_Negative(result.ptr())) : result;
}

nb::object convert_value(CellDecoder& decoder, duckdb_vector vector, idx_t row);

// Convert one cell of `vector`, whose data and validity are passed in.
nb::object convert_cell(CellDecoder& decoder, duckdb_vector vector, void* data, uint64_t* validity,
                        idx_t row) {
    if (validity && !duckdb_validity_row_is_valid(validity, row)) return nb::none();

    switch (decoder.type) {
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
            return decimal_to_py(data, row, decoder.internal, decoder.scale);
        case DUCKDB_TYPE_VARCHAR: {
            auto* s = (duckdb_string_t*)data;
            return nb::str(duckdb_string_t_data(&s[row]), duckdb_string_t_length(s[row]));
        }
        case DUCKDB_TYPE_BLOB:
        case DUCKDB_TYPE_GEOMETRY: {  // GEOMETRY is stored as WKB
            auto* s = (duckdb_string_t*)data;
            return nb::bytes(duckdb_string_t_data(&s[row]), duckdb_string_t_length(s[row]));
        }
        case DUCKDB_TYPE_BIT:
            return bit_to_py(((duckdb_string_t*)data)[row]);
        case DUCKDB_TYPE_BIGNUM:
            return bignum_to_py(((duckdb_string_t*)data)[row]);
        case DUCKDB_TYPE_DATE:
            return date_to_py(((int32_t*)data)[row]);
        case DUCKDB_TYPE_TIME:
            return time_to_py(((int64_t*)data)[row]);
        case DUCKDB_TYPE_TIME_NS:
            return time_to_py(floor_div(((int64_t*)data)[row], 1000));
        case DUCKDB_TYPE_TIME_TZ:
            return time_tz_to_py(((uint64_t*)data)[row], decoder.timezones);
        case DUCKDB_TYPE_TIMESTAMP:
            return timestamp_to_py(((int64_t*)data)[row], false);
        case DUCKDB_TYPE_TIMESTAMP_TZ:
            return timestamp_to_py(((int64_t*)data)[row], true);
        case DUCKDB_TYPE_TIMESTAMP_S: {
            int64_t v = ((int64_t*)data)[row];
            return scaled_timestamp_to_py(v, 1000000,
                                          duckdb_is_finite_timestamp_s(duckdb_timestamp_s{v}));
        }
        case DUCKDB_TYPE_TIMESTAMP_MS: {
            int64_t v = ((int64_t*)data)[row];
            return scaled_timestamp_to_py(v, 1000,
                                          duckdb_is_finite_timestamp_ms(duckdb_timestamp_ms{v}));
        }
        case DUCKDB_TYPE_TIMESTAMP_NS:
            return timestamp_ns_to_py(((int64_t*)data)[row]);
        case DUCKDB_TYPE_UUID:
            return uuid_to_py(((duckdb_hugeint*)data)[row]);
        case DUCKDB_TYPE_INTERVAL:
            return interval_to_py(((duckdb_interval*)data)[row]);
        case DUCKDB_TYPE_SQLNULL:
            return nb::none();
        case DUCKDB_TYPE_ENUM: {
            idx_t index = enum_index(data, row, decoder.internal);
            if (index >= decoder.enum_values.size()) {
                throw DuckyError("ducky: ENUM index out of range");
            }
            return decoder.enum_values[index];
        }
        case DUCKDB_TYPE_LIST: {
            duckdb_list_entry entry = ((duckdb_list_entry*)data)[row];
            duckdb_vector child = duckdb_list_vector_get_child(vector);
            void* child_data = duckdb_vector_get_data(child);
            uint64_t* child_validity = duckdb_vector_get_validity(child);
            nb::object items = steal_checked(PyList_New((Py_ssize_t)entry.length));
            for (idx_t k = 0; k < entry.length; ++k) {
                nb::object item = convert_cell(decoder.children[0], child, child_data,
                                               child_validity, entry.offset + k);
                PyList_SET_ITEM(items.ptr(), (Py_ssize_t)k, item.release().ptr());
            }
            return items;
        }
        case DUCKDB_TYPE_ARRAY: {
            idx_t size = decoder.array_size;
            duckdb_vector child = duckdb_array_vector_get_child(vector);
            void* child_data = duckdb_vector_get_data(child);
            uint64_t* child_validity = duckdb_vector_get_validity(child);
            nb::object items = steal_checked(PyList_New((Py_ssize_t)size));
            for (idx_t k = 0; k < size; ++k) {
                nb::object item = convert_cell(decoder.children[0], child, child_data,
                                               child_validity, row * size + k);
                PyList_SET_ITEM(items.ptr(), (Py_ssize_t)k, item.release().ptr());
            }
            return items;
        }
        case DUCKDB_TYPE_STRUCT: {
            nb::dict fields;
            for (idx_t c = 0; c < (idx_t)decoder.children.size(); ++c) {
                duckdb_vector child = duckdb_struct_vector_get_child(vector, c);
                fields[decoder.names[c]] = convert_value(decoder.children[c], child, row);
            }
            return fields;
        }
        case DUCKDB_TYPE_UNION: {
            // Physical layout: STRUCT(tag UTINYINT, member_0, member_1, ...);
            // the value is the member the tag selects.
            duckdb_vector tags = duckdb_struct_vector_get_child(vector, 0);
            uint8_t tag = ((uint8_t*)duckdb_vector_get_data(tags))[row];
            if (tag >= decoder.children.size()) throw DuckyError("ducky: UNION tag out of range");
            duckdb_vector member = duckdb_struct_vector_get_child(vector, (idx_t)tag + 1);
            return convert_value(decoder.children[tag], member, row);
        }
        case DUCKDB_TYPE_MAP: {
            // Physical layout: LIST(STRUCT(key, value)).
            duckdb_list_entry entry = ((duckdb_list_entry*)data)[row];
            duckdb_vector pairs = duckdb_list_vector_get_child(vector);
            duckdb_vector keys = duckdb_struct_vector_get_child(pairs, 0);
            duckdb_vector values = duckdb_struct_vector_get_child(pairs, 1);
            void* key_data = duckdb_vector_get_data(keys);
            uint64_t* key_validity = duckdb_vector_get_validity(keys);
            void* value_data = duckdb_vector_get_data(values);
            uint64_t* value_validity = duckdb_vector_get_validity(values);
            nb::dict mapping;
            for (idx_t k = 0; k < entry.length; ++k) {
                mapping[convert_cell(decoder.children[0], keys, key_data, key_validity,
                                     entry.offset + k)] =
                    convert_cell(decoder.children[1], values, value_data, value_validity,
                                 entry.offset + k);
            }
            return mapping;
        }
        case DUCKDB_TYPE_VARIANT:
            throw DuckyError(
                "ducky: VARIANT values are not decoded to Python; cast the column (e.g. "
                "CAST(v AS JSON) or v::VARCHAR) or use .arrow()");
        default:
            throw DuckyError(std::string("ducky: column type ") + duckdb_type_name(decoder.type) +
                             " is not decoded yet");
    }
}

// Convert one cell of `vector`, resolving its data and validity.
nb::object convert_value(CellDecoder& decoder, duckdb_vector vector, idx_t row) {
    return convert_cell(decoder, vector, duckdb_vector_get_data(vector),
                        duckdb_vector_get_validity(vector), row);
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

// Arrow stream state for a Result. Callbacks may run on any thread, with or
// without the GIL; they only touch the Result through its cursor lock.
struct ArrowStreamState {
    nb::object owner;                     // keeps the Result (and its duckdb_result) alive
    Result* result;                       // borrowed from owner
    duckdb_arrow_options options;         // owned by the stream, destroyed on release
    bool fast;                            // all columns eligible -> one hand-built batch
    bool done;                            // fast path emits exactly one combined batch
    duckdb_data_chunk scratch = nullptr;  // reused across get_next calls
    std::vector<size_t> widths;           // per-column element width (fast path only)
    std::string last_error;
};

// Scope owner for a result-shaped chunk.
struct ScratchChunk {
    duckdb_data_chunk chunk;
    explicit ScratchChunk(duckdb_result* result) : chunk(make_result_chunk(result)) {}
    ~ScratchChunk() {
        if (chunk) duckdb_destroy_data_chunk(&chunk);
    }
    ScratchChunk(const ScratchChunk&) = delete;
    ScratchChunk& operator=(const ScratchChunk&) = delete;
};

// The unconsumed chunks of a result, retained when the row count is unknown.
struct ChunkList {
    std::vector<duckdb_data_chunk> chunks;
    std::vector<idx_t> sizes;
    idx_t total = 0;

    // Requires `result`'s cursor lock.
    explicit ChunkList(Result& result) {
        for (;;) {
            duckdb_data_chunk ch = make_result_chunk(result.raw_result());
            if (!ch) return;
            bool fetched;
            try {
                fetched = result.next_chunk_locked(ch);
            } catch (...) {
                duckdb_destroy_data_chunk(&ch);
                throw;
            }
            if (!fetched) {
                duckdb_destroy_data_chunk(&ch);
                return;
            }
            idx_t sz = duckdb_data_chunk_get_size(ch);
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
    idx_t r = 0;
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    // DuckDB's mask is LSB-first uint64 words, so on little-endian machines
    // its bytes are Arrow bitmap bytes: copy the whole ones (every chunk but
    // a short last one starts byte-aligned).
    if (row_off % 8 == 0) {
        r = sz / 8 * 8;
        std::memcpy(validity + row_off / 8, val, r / 8);
    }
#endif
    for (; r < sz; ++r) {
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
    duckdb_result* result = state->result->raw_result();
    idx_t n = duckdb_column_count(result);
    std::vector<duckdb_logical_type> types(n);
    std::vector<const char*> names(n);
    for (idx_t i = 0; i < n; ++i) {
        types[i] = duckdb_column_logical_type(result, i);
        names[i] = duckdb_column_name(result, i);
    }
    duckdb_error_data error =
        duckdb_to_arrow_schema(state->options, types.data(), names.data(), n, out);
    for (idx_t i = 0; i < n; ++i) duckdb_destroy_logical_type(&types[i]);
    return consume_error(state, error) ? EIO : 0;
}

// Build one combined batch from exactly `total` remaining rows.
void build_fast_batch_known_total(ArrowStreamState* state, ArrowArray* out, idx_t total) {
    Result& result = *state->result;
    idx_t ncols = duckdb_column_count(result.raw_result());
    size_t valid_bytes = (total + 7) / 8;

    auto b = std::make_unique<FastBatch>(ncols);
    std::vector<uint8_t*> validities(ncols, nullptr);
    for (idx_t c = 0; c < ncols; ++c) {
        b->child_bufs[c] = {nullptr, b->alloc(total * state->widths[c])};
    }

    ScratchChunk scratch(result.raw_result());
    if (!scratch.chunk) throw std::bad_alloc();
    size_t row_off = 0;
    while (result.next_chunk_locked(scratch.chunk)) {
        idx_t sz = duckdb_data_chunk_get_size(scratch.chunk);
        if (row_off + sz > total) {
            throw DuckyError("ducky: result produced more rows than expected");
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
        throw DuckyError("ducky: result produced fewer rows than expected");
    }

    finish_fast_batch(std::move(b), out, total, ncols, validities);
}

// Drain layout-compatible columns into one Arrow batch. Requires the cursor lock.
void build_fast_batch(ArrowStreamState* state, ArrowArray* out) {
    // The remaining count is taken now, under the cursor lock, so rows other
    // consumers pulled since the capsule was created are accounted for.
    if (std::optional<idx_t> remaining = state->result->remaining_rows_locked()) {
        if (*remaining == 0) return;  // no rows -> emit no batch (out->release stays nullptr)
        build_fast_batch_known_total(state, out, *remaining);
        return;
    }
    idx_t ncols = duckdb_column_count(state->result->raw_result());

    ChunkList pulled(*state->result);
    if (pulled.total == 0) return;
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
    // Nothing below touches Python objects; drop the GIL (if the consumer
    // holds it) before waiting on the cursor, and keep it dropped while DuckDB
    // executes, so workers running Python callbacks can take it.
    release_gil_if_held release;
    try {
        std::unique_lock<std::mutex> lock = state->result->lock_cursor();
        if (state->fast) {
            if (state->done) return 0;  // the single combined batch was already emitted
            state->done = true;
            build_fast_batch(state, out);
            return 0;
        }
        // Convert one reusable DuckDB chunk into Arrow-owned buffers.
        if (!state->scratch) {
            state->scratch = make_result_chunk(state->result->raw_result());
            if (!state->scratch) return 0;
        }
        if (!state->result->next_chunk_locked(state->scratch)) return 0;  // exhausted
        duckdb_error_data error = duckdb_data_chunk_to_arrow(state->options, state->scratch, out);
        return consume_error(state, error) ? EIO : 0;
    } catch (const std::exception& e) {
        if (out->release) out->release(out);
        state->last_error = e.what();
        return EIO;
    }
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

CellDecoder::CellDecoder(duckdb_logical_type logical) : type(duckdb_get_type_id(logical)) {
    // Decode a child type, which the C API hands out owned.
    auto add_child = [this](duckdb_logical_type child) {
        try {
            children.emplace_back(child);
        } catch (...) {
            duckdb_destroy_logical_type(&child);
            throw;
        }
        duckdb_destroy_logical_type(&child);
    };
    switch (type) {
        case DUCKDB_TYPE_DECIMAL:
            internal = duckdb_decimal_internal_type(logical);
            scale = duckdb_decimal_scale(logical);
            break;
        case DUCKDB_TYPE_ENUM: {
            internal = duckdb_enum_internal_type(logical);
            uint32_t n = duckdb_enum_dictionary_size(logical);
            enum_values.reserve(n);
            for (uint32_t i = 0; i < n; ++i) {
                char* value = duckdb_enum_dictionary_value(logical, i);
                nb::str out(value);
                duckdb_free(value);
                enum_values.push_back(std::move(out));
            }
            break;
        }
        case DUCKDB_TYPE_LIST:
            add_child(duckdb_list_type_child_type(logical));
            break;
        case DUCKDB_TYPE_ARRAY:
            array_size = duckdb_array_type_array_size(logical);
            add_child(duckdb_array_type_child_type(logical));
            break;
        case DUCKDB_TYPE_STRUCT: {
            idx_t n = duckdb_struct_type_child_count(logical);
            for (idx_t i = 0; i < n; ++i) {
                char* name = duckdb_struct_type_child_name(logical, i);
                nb::str out(name);
                duckdb_free(name);
                names.push_back(std::move(out));
                add_child(duckdb_struct_type_child_type(logical, i));
            }
            break;
        }
        case DUCKDB_TYPE_MAP:
            add_child(duckdb_map_type_key_type(logical));
            add_child(duckdb_map_type_value_type(logical));
            break;
        case DUCKDB_TYPE_UNION: {
            idx_t n = duckdb_union_type_member_count(logical);
            for (idx_t i = 0; i < n; ++i) add_child(duckdb_union_type_member_type(logical, i));
            break;
        }
        default:
            break;
    }
}

duckdb_data_chunk make_result_chunk(duckdb_result* result) {
    idx_t n = duckdb_column_count(result);
    if (n == 0) return nullptr;
    std::vector<duckdb_logical_type> types(n);
    for (idx_t i = 0; i < n; ++i) types[i] = duckdb_column_logical_type(result, i);
    duckdb_data_chunk chunk = duckdb_create_data_chunk(types.data(), n);
    for (idx_t i = 0; i < n; ++i) duckdb_destroy_logical_type(&types[i]);
    return chunk;
}

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
    state->result = this;
    state->options = options;
    state->fast = fast;
    state->done = false;
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
    streaming_ = duckdb_result_is_streaming(result_);
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
    // Destroying an unfinished streaming result cleans up its query under the
    // client context lock, which a query running Python callbacks on another
    // thread may hold while it waits for the GIL.
    release_gil_if_held release;
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

std::unique_lock<std::mutex> Result::lock_cursor() { return lock_releasing_gil(mu_); }

bool Result::fetch_into_locked(duckdb_data_chunk dst) {
    // The end is sticky: fetching past it would fail on a closed stream.
    if (exhausted_) return false;
    bool fetched;
    {
        // A fetch can drive query execution (streaming results); DuckDB workers
        // running Python UDFs or Arrow scans need the GIL meanwhile.
        release_gil_if_held release;
        fetched = duckdb_fetch_chunk_into(result_, dst);
    }
    if (fetched) return true;
    // A false return is either the end of the result or a failure that DuckDB
    // records on the result (execution errors, interrupts).
    if (const char* error = duckdb_result_error(&result_)) throw DuckyError(error);
    exhausted_ = true;
    return false;
}

bool Result::next_chunk_locked(duckdb_data_chunk dst) {
    if (chunk_ && cursor_ < chunk_size_) {
        // Hand out the rows the row cursor buffered but has not returned yet.
        idx_t count = chunk_size_ - cursor_;
        duckdb_data_chunk_reset(dst);
        duckdb_selection_vector sel = duckdb_create_selection_vector(chunk_size_);
        sel_t* index = duckdb_selection_vector_get_data_ptr(sel);
        for (idx_t i = 0; i < chunk_size_; ++i) index[i] = (sel_t)i;
        for (idx_t c = 0; c < column_count_; ++c) {
            duckdb_vector_copy_sel(duckdb_data_chunk_get_vector(chunk_, c),
                                   duckdb_data_chunk_get_vector(dst, c), sel, chunk_size_, cursor_,
                                   0);
        }
        duckdb_destroy_selection_vector(sel);
        duckdb_data_chunk_set_size(dst, count);
        release_chunk();
        rows_returned_ += count;
        return true;
    }
    release_chunk();
    for (;;) {
        if (!fetch_into_locked(dst)) return false;
        idx_t size = duckdb_data_chunk_get_size(dst);
        if (size == 0) continue;
        rows_returned_ += size;
        return true;
    }
}

std::optional<idx_t> Result::remaining_rows_locked() {
    if (streaming_) return std::nullopt;
    idx_t total = duckdb_row_count(&result_);
    return total > rows_returned_ ? total - rows_returned_ : 0;
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

// Requires the cursor lock.
bool Result::ensure_row() {
    while (!chunk_ || cursor_ >= chunk_size_) {
        // Refill the cursor's scratch chunk in place.
        if (!chunk_) {
            chunk_ = make_result_chunk(&result_);
            if (!chunk_) return false;  // zero-column result
        }
        if (!fetch_into_locked(chunk_)) {
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

void Result::ensure_decoders() {
    if (decoders_.size() == column_count_) return;
    decoders_.clear();
    decoders_.reserve(column_count_);
    for (duckdb_logical_type logical : column_types_) decoders_.emplace_back(logical);
}

// Requires the cursor lock.
nb::object Result::build_row() {
    ensure_decoders();
    idx_t row = cursor_++;
    rows_returned_++;
    // Own the partial tuple across conversion errors.
    nb::object tuple = steal_checked(PyTuple_New((Py_ssize_t)column_count_));
    for (idx_t c = 0; c < column_count_; ++c) {
        const ColumnCursor& cur = cursors_[c];
        nb::object value = convert_cell(decoders_[c], cur.vector, cur.data, cur.validity, row);
        PyTuple_SET_ITEM(tuple.ptr(), (Py_ssize_t)c, value.release().ptr());
    }
    return tuple;
}

nb::object Result::fetchone() {
    auto lock = lock_cursor();
    if (!ensure_row()) return nb::none();
    return build_row();
}

nb::list Result::fetchmany(int64_t size) {
    auto lock = lock_cursor();
    nb::list out;
    for (int64_t i = 0; i < size && ensure_row(); ++i) out.append(build_row());
    return out;
}

nb::list Result::fetchall() {
    auto lock = lock_cursor();
    // Pre-size for the known unconsumed materialized rows.
    std::optional<idx_t> remaining = remaining_rows_locked();
    if (remaining && *remaining > 0) {
        PyObject* list = PyList_New((Py_ssize_t)*remaining);
        if (!list) throw nb::python_error();
        nb::list out = nb::steal<nb::list>(list);
        for (idx_t i = 0; i < *remaining; ++i) {
            if (!ensure_row()) {  // defensive; a materialized count is exact
                if (PyList_SetSlice(list, (Py_ssize_t)i, (Py_ssize_t)*remaining, nullptr) < 0) {
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
    auto lock = lock_cursor();
    if (!ensure_row()) {
        throw DuckyError("ducky: fetchitem() requires exactly 1 row; the result is empty");
    }
    // Decode then confirm no second row remains. value is materialized, so it
    // survives the chunk release that the follow-up ensure_row() may trigger.
    ensure_decoders();
    idx_t row = cursor_++;
    rows_returned_++;
    const ColumnCursor& cur = cursors_[0];
    nb::object value = convert_cell(decoders_[0], cur.vector, cur.data, cur.validity, row);
    if (ensure_row()) {
        throw DuckyError("ducky: fetchitem() requires exactly 1 row; the result has more");
    }
    return value;
}

int64_t Result::rows_changed() {
    // Streaming results can't report the count without materializing.
    if (streaming_) return 0;
    return (int64_t)duckdb_rows_changed(&result_);
}

duckdb_data_chunk Result::fetch_raw_chunk_locked() {
    duckdb_data_chunk chunk = chunk_pool_->take();
    if (!chunk) chunk = make_result_chunk(&result_);
    if (!chunk) return nullptr;  // zero-column result
    bool fetched;
    try {
        fetched = next_chunk_locked(chunk);
    } catch (...) {
        duckdb_destroy_data_chunk(&chunk);
        throw;
    }
    if (!fetched) {
        duckdb_destroy_data_chunk(&chunk);
        return nullptr;
    }
    return chunk;
}

nb::object Result::fetch_chunk() {
    auto lock = lock_cursor();
    duckdb_data_chunk chunk = fetch_raw_chunk_locked();
    if (!chunk) return nb::none();
    // The Python wrapper owns the allocated Chunk.
    return nb::cast(new Chunk(chunk, schema_, handle_, chunk_pool_), nb::rv_policy::take_ownership);
}

namespace {

// NumPy output layout resolved before scanning.
struct NumpyColumnPlan {
    size_t elem_size;
    nb::dlpack::dtype dtype;
    nb::object struct_view;  // non-null → reinterpret via .view()
    duckdb_type type;
};

NumpyColumnPlan plan_column(duckdb_type tid, duckdb_logical_type logical, const std::string& name) {
    if (const TypeSpec* spec = typespec_for(tid)) {
        return NumpyColumnPlan{spec->size, spec->dtype(), nb::object(), tid};
    }
    const StructDtypes& sd = struct_dtypes();
    switch (tid) {
        case DUCKDB_TYPE_HUGEINT:
            return NumpyColumnPlan{sizeof(duckdb_hugeint), nb::dtype<uint8_t>(), sd.hugeint, tid};
        case DUCKDB_TYPE_UHUGEINT:
            return NumpyColumnPlan{sizeof(duckdb_uhugeint), nb::dtype<uint8_t>(), sd.uhugeint, tid};
        case DUCKDB_TYPE_INTERVAL:
            return NumpyColumnPlan{sizeof(duckdb_interval), nb::dtype<uint8_t>(), sd.interval, tid};
        case DUCKDB_TYPE_DECIMAL: {
            duckdb_type internal = duckdb_decimal_internal_type(logical);
            if (internal == DUCKDB_TYPE_HUGEINT) {
                return NumpyColumnPlan{sizeof(duckdb_hugeint), nb::dtype<uint8_t>(), sd.hugeint,
                                       tid};
            }
            const TypeSpec* spec = typespec_for(internal);  // always primitive here
            return NumpyColumnPlan{spec->size, spec->dtype(), nb::object(), tid};
        }
        default:
            throw DuckyError(std::string("ducky: to_numpy: column '") + name + "' has type " +
                             duckdb_type_name(tid) +
                             " (no flat ndarray representation; "
                             "use .arrow() for VARCHAR, LIST, STRUCT, MAP, ...)");
    }
}

// Copy `count` values of one column into `dst`. NULLs become NaN in FLOAT /
// DOUBLE columns; other dtypes have no missing-value representation, so NULLs
// raise rather than surfacing whatever bytes DuckDB left in the slot.
void copy_numpy_values(char* dst, duckdb_vector v, idx_t count, const NumpyColumnPlan& plan,
                       const std::string& name) {
    std::memcpy(dst, duckdb_vector_get_data(v), count * plan.elem_size);
    uint64_t* validity = duckdb_vector_get_validity(v);
    if (!validity) return;
    for (idx_t r = 0; r < count; ++r) {
        if (duckdb_validity_row_is_valid(validity, r)) continue;
        if (plan.type == DUCKDB_TYPE_DOUBLE) {
            ((double*)dst)[r] = std::numeric_limits<double>::quiet_NaN();
        } else if (plan.type == DUCKDB_TYPE_FLOAT) {
            ((float*)dst)[r] = std::numeric_limits<float>::quiet_NaN();
        } else {
            throw DuckyError(std::string("ducky: to_numpy: column '") + name + "' (" +
                             duckdb_type_name(plan.type) +
                             ") contains NULLs, which its NumPy dtype cannot represent; use "
                             "iter_batches(with_validity=True), .arrow(), or COALESCE in SQL");
        }
    }
}

}  // namespace

nb::object Result::fetch_numpy_chunk() {
    // Validate every column before pulling, so an unsupported type doesn't
    // consume (and lose) a chunk.
    for (idx_t i = 0; i < column_count_; ++i) {
        plan_column(types_[i], column_types_[i], schema_->names[i]);
    }
    duckdb_data_chunk chunk;
    {
        auto lock = lock_cursor();
        chunk = fetch_raw_chunk_locked();
    }
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

    // Wrap a filled buffer as the column's ndarray.
    auto wrap = [&](idx_t c, char* buf, idx_t total, nb::object owner) -> nb::object {
        const NumpyColumnPlan& p = plans[c];
        if (p.struct_view) {
            // Wrap as a flat uint8 buffer (total * elem_size bytes) and let
            // numpy reinterpret it as a structured array of `total` items.
            size_t byte_shape[1] = {total * p.elem_size};
            nb::ndarray<nb::numpy> raw(buf, 1, byte_shape, owner, nullptr, p.dtype);
            return nb::cast(raw).attr("view")(p.struct_view);
        }
        size_t shape[1] = {(size_t)total};
        return nb::cast(nb::ndarray<nb::numpy>(buf, 1, shape, owner, nullptr, p.dtype));
    };

    auto lock = lock_cursor();

    // Allocate upfront when the exact remaining row count is known.
    if (std::optional<idx_t> remaining = remaining_rows_locked()) {
        idx_t total = *remaining;
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
            while (next_chunk_locked(scratch.chunk)) {
                idx_t sz = duckdb_data_chunk_get_size(scratch.chunk);
                if (offset + sz > total) {
                    throw DuckyError("ducky: to_numpy: result produced more rows than expected");
                }
                for (idx_t c = 0; c < column_count_; ++c) {
                    if (!selected[c]) continue;
                    copy_numpy_values(bufs[c] + offset * plans[c].elem_size,
                                      duckdb_data_chunk_get_vector(scratch.chunk, c), sz, plans[c],
                                      schema_->names[c]);
                }
                offset += sz;
            }
            if (offset != total) {
                throw DuckyError("ducky: to_numpy: result produced fewer rows than expected");
            }
        }

        nb::dict out;
        for (idx_t c = 0; c < column_count_; ++c) {
            if (!selected[c]) continue;
            out[schema_->names[c].c_str()] = wrap(c, bufs[c], total, owners[c]);
        }
        return out;
    }

    // Retain chunks when the total must be discovered first.
    ChunkList pulled(*this);
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
            copy_numpy_values((char*)buf + offset, pulled.vector(i, c), pulled.sizes[i], p,
                              schema_->names[c]);
            offset += pulled.sizes[i] * p.elem_size;
        }
        out[schema_->names[c].c_str()] = wrap(c, (char*)buf, total, owner);
    }
    return out;
}
