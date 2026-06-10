#include "result.hpp"

#include <datetime.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cerrno>
#include <cstdlib>

#include "arrow_abi.h"
#include "chunk.hpp"
#include "ducky.hpp"

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

// ArrowArrayStream backing for a duckdb_result: pulls chunks on demand and
// exports each via the DuckDB Arrow C API. `owner` keeps the Result alive
// for the stream's lifetime.
struct ArrowStreamState {
    nb::object owner;              // keeps the Result (and its DuckDBHandle) alive
    duckdb_result* result;         // borrowed from owner
    duckdb_arrow_options options;  // owned by the stream, destroyed on release
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

int stream_get_next(ArrowArrayStream* stream, ArrowArray* out) {
    auto* state = (ArrowStreamState*)stream->private_data;
    out->release = nullptr;  // Arrow signals end-of-stream with a released array.
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
    // Options carry a live client context from the connection; the stream
    // owns them and frees them on release.
    duckdb_arrow_options options = nullptr;
    duckdb_connection_get_arrow_options(handle_->connection, &options);

    auto* stream = (ArrowArrayStream*)calloc(1, sizeof(ArrowArrayStream));
    if (!stream) throw std::bad_alloc();
    auto* state = new ArrowStreamState{self, &result_, options, std::string()};
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
