#include "result.hpp"

#include <datetime.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>

#include "arrow_abi.h"
#include "chunk.hpp"
#include "ducky.hpp"
#include "function.hpp"

using namespace nb::literals;

namespace {

// Python types and instances used by convert_value. Imported once,
// intentionally leaked so destructors don't run at interpreter shutdown.
// date / time / timedelta values are built via the PyDateTime_* CAPI macros.
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

nb::object date_to_py(int32_t days) {
    duckdb_date_struct s = duckdb_from_date(duckdb_date{days});
    return nb::steal(PyDate_FromDate(s.year, s.month, s.day));
}

nb::object time_to_py(int64_t micros) {
    duckdb_time_struct s = duckdb_from_time(duckdb_time{micros});
    return nb::steal(PyTime_FromTime(s.hour, s.min, s.sec, s.micros));
}

nb::object timestamp_to_py(int64_t micros, bool utc) {
    duckdb_timestamp_struct s = duckdb_from_timestamp(duckdb_timestamp{micros});
    if (utc) {
        // No CAPI overload takes tzinfo; route through the Python constructor.
        return py_types().datetime(s.date.year, s.date.month, s.date.day, s.time.hour, s.time.min,
                                   s.time.sec, s.time.micros, py_types().timezone_utc);
    }
    return nb::steal(PyDateTime_FromDateAndTime(s.date.year, s.date.month, s.date.day, s.time.hour,
                                                s.time.min, s.time.sec, s.time.micros));
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
    // timedelta has no months; approximate 1 month as 30 days. Use the Arrow
    // path for an exact month / day / micros interval.
    int days = value.days + value.months * 30;
    return nb::steal(PyDelta_FromDSU(days, 0, value.micros));
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

// Recursively convert the value at `row` of `vector` to a Python object,
// descending into LIST/STRUCT/ARRAY/MAP children. `logical` is borrowed: top-
// level columns pass the Result's cached types so the hot path never allocates
// per cell; nested children fetch and free their own per cell.
nb::object convert_value(duckdb_vector vector, duckdb_logical_type logical, idx_t row) {
    uint64_t* validity = duckdb_vector_get_validity(vector);
    if (validity && !duckdb_validity_row_is_valid(validity, row)) return nb::none();

    void* data = duckdb_vector_get_data(vector);
    switch (duckdb_get_type_id(logical)) {
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
            throw DuckyError(std::string("ducky: column type ") +
                             duckdb_type_name(duckdb_get_type_id(logical)) + " is not decoded yet");
    }
}

// Byte width of a fixed-width column type whose DuckDB in-memory layout is
// bit-identical to its Arrow primitive layout, or 0 if the type is not eligible
// for the combined fast path. Eligible types are the ones whose flat DuckDB
// buffer concatenates straight into Arrow's data buffer with no conversion: the
// fixed-width numerics, plus the temporals whose storage matches Arrow's unit 1:1
// (DATE=date32 days, TIME=time64[us], TIMESTAMP[_S/MS/NS/TZ]=timestamp of the same
// unit — DuckDB stores and Arrow expects the identical native-unit int), each
// verified bit-equal to duckdb_data_chunk_to_arrow in test_arrow.py. Excludes
// BOOLEAN (Arrow packs it 1 bit/value), HUGEINT/DECIMAL/INTERVAL, and all
// variable-length / nested types — those take the per-chunk fallback below.
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

// ArrowArrayStream backing for a duckdb_result. The schema always comes from the
// C API (duckdb_to_arrow_schema). get_next has two modes: when every column is a
// fast-path-eligible fixed-width numeric, it drains the whole result and
// hand-builds one combined Arrow record batch (concatenated data buffers +
// validity bitmaps); otherwise it falls back to one Arrow array per DuckDB chunk
// via duckdb_data_chunk_to_arrow, which handles every type. `owner` keeps the
// ducky Result (and its duckdb_result) alive for the stream's lifetime.
struct ArrowStreamState {
    nb::object owner;              // keeps the Result (and its duckdb_result) alive
    duckdb_result* result;         // borrowed from owner
    duckdb_arrow_options options;  // owned by the stream, destroyed on release
    bool fast;                     // all columns eligible -> one hand-built batch
    bool done;                     // fast path emits exactly one combined batch
    std::vector<size_t> widths;    // per-column element width (fast path only)
    std::string last_error;
};

// Record + destroy a duckdb_error_data; returns true when an error was present.
bool consume_error(ArrowStreamState* state, duckdb_error_data error) {
    if (!error) return false;
    bool has_error = duckdb_error_data_has_error(error);
    if (has_error) state->last_error = duckdb_error_data_message(error);
    duckdb_destroy_error_data(&error);
    return has_error;
}

// Owns every heap allocation backing one hand-built record batch. The root
// ArrowArray's release frees this; children carry a no-op release because the
// importer only releases the root, which tears down the whole tree at once.
struct FastBatch {
    std::vector<void*> allocations;                      // data + validity buffers to free
    std::vector<ArrowArray> children;                    // one struct per column (stable)
    std::vector<ArrowArray*> child_ptrs;                 // root.children points here
    std::vector<std::array<const void*, 2>> child_bufs;  // [validity, data] per column
    const void* root_buf[1] = {nullptr};                 // struct array: one (null) validity buf
};

void fast_release_child(ArrowArray* a) { a->release = nullptr; }  // root owns the memory

void fast_release_root(ArrowArray* a) {
    auto* b = (FastBatch*)a->private_data;
    for (void* p : b->allocations) std::free(p);
    delete b;
    a->release = nullptr;
    a->private_data = nullptr;
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

// Drain the whole result and hand-build one combined Arrow record batch. Only
// called when every column is fast_arrow_width-eligible. Sets out->release to
// nullptr (end-of-stream) for an empty result. Throws on allocation failure.
void build_fast_batch(ArrowStreamState* state, ArrowArray* out) {
    idx_t ncols = duckdb_column_count(state->result);

    std::vector<duckdb_data_chunk> chunks;
    std::vector<idx_t> sizes;
    idx_t total = 0;
    while (true) {
        duckdb_data_chunk ch = duckdb_fetch_chunk(*state->result);
        if (!ch) break;
        idx_t sz = duckdb_data_chunk_get_size(ch);
        if (sz == 0) {
            duckdb_destroy_data_chunk(&ch);
            continue;
        }
        chunks.push_back(ch);
        sizes.push_back(sz);
        total += sz;
    }
    auto cleanup_chunks = [&] {
        for (duckdb_data_chunk ch : chunks) duckdb_destroy_data_chunk(&ch);
    };
    if (total == 0) {  // no rows -> emit no batch (out->release stays nullptr)
        cleanup_chunks();
        return;
    }

    auto* b = new FastBatch();
    b->children.resize(ncols);
    b->child_ptrs.resize(ncols);
    b->child_bufs.resize(ncols);
    size_t valid_bytes = (total + 7) / 8;

    try {
        for (idx_t c = 0; c < ncols; ++c) {
            size_t width = state->widths[c];
            void* data = std::malloc(total * width);
            if (!data) throw std::bad_alloc();
            b->allocations.push_back(data);

            // Concatenate each chunk's flat data buffer; build a validity bitmap
            // lazily and per-row (so it is correct regardless of chunk sizes) the
            // first time a column actually carries nulls. No nulls anywhere -> no
            // validity buffer at all, matching Arrow's all-valid convention.
            uint8_t* validity = nullptr;
            size_t row_off = 0;
            for (size_t i = 0; i < chunks.size(); ++i) {
                duckdb_vector v = duckdb_data_chunk_get_vector(chunks[i], (idx_t)c);
                std::memcpy((char*)data + row_off * width, duckdb_vector_get_data(v),
                            sizes[i] * width);
                uint64_t* val = duckdb_vector_get_validity(v);
                if (val) {
                    if (!validity) {
                        validity = (uint8_t*)std::malloc(valid_bytes);
                        if (!validity) throw std::bad_alloc();
                        b->allocations.push_back(validity);
                        std::memset(validity, 0xFF, valid_bytes);  // start all-valid
                    }
                    for (idx_t r = 0; r < sizes[i]; ++r) {
                        if (!duckdb_validity_row_is_valid(val, r)) {
                            size_t bit = row_off + r;
                            validity[bit / 8] &= (uint8_t)~(1u << (bit % 8));
                        }
                    }
                }
                row_off += sizes[i];
            }

            b->child_bufs[c] = {validity, data};
            ArrowArray& child = b->children[c];
            child = {};
            child.length = (int64_t)total;
            child.null_count = validity ? -1 : 0;  // -1: importer counts lazily
            child.n_buffers = 2;
            child.buffers = b->child_bufs[c].data();
            child.release = fast_release_child;
            b->child_ptrs[c] = &child;
        }
    } catch (...) {
        cleanup_chunks();
        for (void* p : b->allocations) std::free(p);
        delete b;
        throw;
    }
    cleanup_chunks();

    *out = {};
    out->length = (int64_t)total;
    out->n_buffers = 1;  // struct validity buffer, null = all valid
    out->n_children = (int64_t)ncols;
    out->buffers = b->root_buf;
    out->children = b->child_ptrs.data();
    out->release = fast_release_root;
    out->private_data = b;
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
    // Fallback: one Arrow array per DuckDB data chunk (correct for every type).
    duckdb_data_chunk chunk = duckdb_fetch_chunk(*state->result);
    if (!chunk) return 0;  // exhausted
    duckdb_error_data error = duckdb_data_chunk_to_arrow(state->options, chunk, out);
    duckdb_destroy_data_chunk(&chunk);
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

nb::object Result::arrow_c_stream(nb::object self) {
    if (!handle_ || !handle_->connection) {
        throw DuckyError("ducky: Arrow export is unavailable; the connection is closed");
    }
    // Options for the C-API schema path (and the per-chunk fallback). Taken from
    // the result so the schema matches the arrays we produce. Owned by the stream.
    duckdb_arrow_options options = duckdb_result_get_arrow_options(&result_);

    // Pick the fast path only if every column's DuckDB layout is bit-identical to
    // its Arrow layout, so the combined batch can be a raw buffer concatenation.
    std::vector<size_t> widths(column_count_);
    bool fast = column_count_ > 0;
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
        release_chunk();
        duckdb_data_chunk chunk = duckdb_fetch_chunk(result_);
        if (!chunk) return false;  // no more chunks
        idx_t size = duckdb_data_chunk_get_size(chunk);
        if (size == 0) {
            duckdb_destroy_data_chunk(&chunk);
            continue;
        }
        chunk_ = chunk;
        chunk_size_ = size;
        cursor_ = 0;
        vectors_.resize(column_count_);
        for (idx_t i = 0; i < column_count_; ++i) {
            vectors_[i] = duckdb_data_chunk_get_vector(chunk_, i);
        }
    }
    return true;
}

nb::object Result::build_row() {
    idx_t row = cursor_++;
    PyObject* tuple = PyTuple_New((Py_ssize_t)column_count_);
    if (!tuple) throw nb::python_error();
    for (idx_t c = 0; c < column_count_; ++c) {
        nb::object value = convert_value(vectors_[c], column_types_[c], row);
        PyTuple_SET_ITEM(tuple, (Py_ssize_t)c, value.release().ptr());
    }
    return nb::steal(tuple);
}

// The cursor methods below mutate chunk_/cursor_/vectors_; nb::lock_self() in
// ducky.cpp serializes concurrent calls on free-threaded builds.
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
    nb::object value = convert_value(vectors_[0], column_types_[0], row);
    if (ensure_row()) {
        throw DuckyError("ducky: fetchitem() requires exactly 1 row; the result has more");
    }
    return value;
}

nb::object Result::fetch_chunk() {
    duckdb_data_chunk chunk = duckdb_fetch_chunk(result_);
    if (!chunk) return nb::none();
    return nb::cast(new Chunk(chunk, schema_, handle_));
}

namespace {

// Per-column plan resolved upfront by to_numpy_native. `elem_size` is the raw
// byte width DuckDB lays out in memory; `dtype` describes how to surface it.
// If `struct_view` is non-null we allocate a uint8 buffer of `total*elem_size`
// bytes and finish with `arr.view(struct_view)` to reinterpret as a structured
// array of `total` items (HUGEINT/UHUGEINT/INTERVAL + DECIMAL/HUGEINT-internal).
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
    // Resolve `columns` (None → all) into a selection mask, and resolve the
    // per-column layout (typespec or struct view). Both run before the scan so
    // we don't waste work on a recoverable type error.
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

    // Pass 1: pull every chunk, recording size; keep them alive for pass 2's
    // memcpy. duckdb_destroy_data_chunk only happens after we've copied out.
    std::vector<duckdb_data_chunk> chunks;
    std::vector<idx_t> sizes;
    idx_t total = 0;
    while (true) {
        duckdb_data_chunk ch = duckdb_fetch_chunk(result_);
        if (!ch) break;
        idx_t sz = duckdb_data_chunk_get_size(ch);
        if (sz == 0) {
            duckdb_destroy_data_chunk(&ch);
            continue;
        }
        chunks.push_back(ch);
        sizes.push_back(sz);
        total += sz;
    }

    auto cleanup_chunks = [&] {
        for (duckdb_data_chunk ch : chunks) duckdb_destroy_data_chunk(&ch);
    };

    nb::dict out;
    try {
        for (idx_t c = 0; c < column_count_; ++c) {
            if (!selected[c]) continue;
            const NumpyColumnPlan& p = plans[c];
            size_t bytes = total * p.elem_size;
            void* buf = bytes ? std::malloc(bytes) : nullptr;
            if (bytes && !buf) throw std::bad_alloc();
            // Capsule frees `buf` when the ndarray's owner ref hits zero.
            nb::capsule owner(buf, [](void* p) noexcept { std::free(p); });

            size_t offset = 0;
            for (size_t i = 0; i < chunks.size(); ++i) {
                duckdb_vector v = duckdb_data_chunk_get_vector(chunks[i], (idx_t)c);
                const void* src = duckdb_vector_get_data(v);
                std::memcpy((char*)buf + offset, src, sizes[i] * p.elem_size);
                offset += sizes[i] * p.elem_size;
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
    } catch (...) {
        cleanup_chunks();
        throw;
    }
    cleanup_chunks();
    return out;
}
