#include "connection.hpp"

#include <datetime.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string_view>

#include "arrow_scan.hpp"
#include "arrow_util.hpp"
#include "ducky.hpp"
#include "prepared.hpp"

namespace {

using namespace nb::literals;

// Cached Python types and 128-bit constants.
struct BindTypes {
    nb::type_object decimal;          // decimal.Decimal
    nb::type_object uuid;             // uuid.UUID
    nb::object epoch_utc;             // datetime(1970, 1, 1, tzinfo=timezone.utc)
    nb::object int128_min;            // -2**127
    nb::object int128_max_plus_one;   // 2**127
    nb::object uint128_max_plus_one;  // 2**128
    nb::object mask64;                // 2**64 - 1
};

const BindTypes& bind_types() {
    static std::atomic<BindTypes*> cached{nullptr};
    static nb::ft_mutex mu;
    return cached_singleton(cached, mu, [] {
        // PyDateTime_IMPORT primes PyDateTimeAPI for this TU.
        if (!PyDateTimeAPI) PyDateTime_IMPORT;
        nb::module_ dt = nb::module_::import_("datetime");
        nb::object utc = dt.attr("timezone").attr("utc");
        nb::object epoch_utc = dt.attr("datetime")(1970, 1, 1, 0, 0, 0, 0, utc);
        nb::object two_pow_127 = nb::int_(1).attr("__lshift__")(127);
        nb::object two_pow_128 = nb::int_(1).attr("__lshift__")(128);
        nb::object mask64 = nb::int_(1).attr("__lshift__")(64).attr("__sub__")(nb::int_(1));
        return BindTypes{
            nb::module_::import_("decimal").attr("Decimal"),
            nb::module_::import_("uuid").attr("UUID"),
            std::move(epoch_utc),
            two_pow_127.attr("__neg__")(),
            std::move(two_pow_127),
            std::move(two_pow_128),
            std::move(mask64),
        };
    });
}

nb::object steal_or_throw(PyObject* obj) {
    if (!obj) throw nb::python_error();
    return nb::steal(obj);
}

// Split a Python int into a 128-bit (lo, hi) pair.
//   is_signed=true  → [-2^127, 2^127 - 1]
//   is_signed=false → [0,      2^128 - 1]
// Anything outside raises DuckyError.
void py_long_to_int128(nb::handle value, bool is_signed, uint64_t& lo, uint64_t& hi) {
    const BindTypes& T = bind_types();
    PyObject* v = value.ptr();
    if (is_signed) {
        if (PyObject_RichCompareBool(v, T.int128_min.ptr(), Py_LT) == 1 ||
            PyObject_RichCompareBool(v, T.int128_max_plus_one.ptr(), Py_GE) == 1) {
            throw DuckyError("ducky: integer overflows HUGEINT (128-bit signed)");
        }
    } else {
        if (PyObject_RichCompareBool(v, nb::int_(0).ptr(), Py_LT) == 1 ||
            PyObject_RichCompareBool(v, T.uint128_max_plus_one.ptr(), Py_GE) == 1) {
            throw DuckyError("ducky: integer overflows UHUGEINT (128-bit unsigned)");
        }
    }

    // Two's-complement of N bits is value mod 2^N; Python's % on negatives
    // already returns the positive residue.
    nb::object wrapped = nb::borrow(value).attr("__mod__")(T.uint128_max_plus_one);
    lo = nb::cast<uint64_t>(wrapped.attr("__and__")(T.mask64));
    hi = nb::cast<uint64_t>(wrapped.attr("__rshift__")(nb::int_(64)));
}

// Convert a tz-aware datetime to UTC microseconds since the unix epoch.
int64_t aware_datetime_to_utc_micros(nb::handle value) {
    nb::object delta = nb::borrow(value) - bind_types().epoch_utc;  // datetime.timedelta
    PyObject* d = delta.ptr();
    int64_t days = PyDateTime_DELTA_GET_DAYS(d);
    int64_t seconds = PyDateTime_DELTA_GET_SECONDS(d);
    int64_t micros = PyDateTime_DELTA_GET_MICROSECONDS(d);
    return ((days * 86400) + seconds) * 1000000 + micros;
}

// A decimal.Decimal's exact DECIMAL(width, scale) form. NaN and infinities
// have none.
std::optional<duckdb_decimal> decimal_parts(nb::handle value, idx_t idx) {
    nb::object parts = value.attr("as_tuple")();
    nb::object exponent = parts.attr("exponent");
    if (!nb::isinstance<nb::int_>(exponent)) return std::nullopt;  // 'n', 'N' or 'F'
    // Build the unscaled integer from the exact digit tuple; Decimal
    // arithmetic would round to the context precision (28 digits by default).
    int64_t exp = nb::cast<int64_t>(exponent);
    std::string digits;
    for (nb::handle d : parts.attr("digits")) digits.push_back((char)('0' + nb::cast<int>(d)));
    if (exp > 0) digits.append((size_t)exp, '0');
    size_t first = digits.find_first_not_of('0');
    int64_t significant = first == std::string::npos ? 1 : (int64_t)(digits.size() - first);
    int64_t scale = exp < 0 ? -exp : 0;
    int64_t width = significant > scale ? significant : (scale > 0 ? scale : 1);
    nb::object unscaled = steal_or_throw(PyLong_FromString(digits.c_str(), nullptr, 10));
    if (nb::cast<int>(parts.attr("sign"))) {
        unscaled = steal_or_throw(PyNumber_Negative(unscaled.ptr()));
    }
    if (width > 38) {
        throw DuckyError("ducky: Decimal parameter at position " + std::to_string(idx) + " needs " +
                         std::to_string(width) + " digits; DuckDB DECIMAL holds at most 38");
    }
    uint64_t lo = 0, hi = 0;
    py_long_to_int128(unscaled, /*is_signed=*/true, lo, hi);
    return duckdb_decimal{(uint8_t)width, (uint8_t)scale, duckdb_hugeint{lo, (int64_t)hi}};
}

// One execution's parameters as C API v2 values, which bind when the
// statement executes.
class V2Parameters {
   public:
    // With `planned`, ints and floats convert to the types its plan assumed,
    // so the execution reuses the plan rather than binding again.
    V2Parameters(duckdb_v2_factory_handle factory, nb::handle parameters,
                 const V2Prepared* planned = nullptr);
    ~V2Parameters() {
        for (duckdb_v2_value_handle& value : values_) duckdb_v2_value_destroy(&value);
    }

    V2Parameters(const V2Parameters&) = delete;
    V2Parameters& operator=(const V2Parameters&) = delete;

    // NULL when binding positionally.
    const char** names() { return names_.empty() ? nullptr : name_ptrs_.data(); }
    const idx_t* name_lengths() { return names_.empty() ? nullptr : name_lengths_.data(); }
    const duckdb_v2_value_handle* values() const { return values_.data(); }
    idx_t count() const { return values_.size(); }

   private:
    // Converts the parameter at position `idx` (1-based). An int becomes the
    // integer type `planned` names if it fits, else a BIGINT; a float becomes
    // a FLOAT if planned so, else a DOUBLE.
    duckdb_v2_value_handle convert(nb::handle value, idx_t idx, DUCKDB_V2_LOGICAL_TYPE_ID planned);
    // Creates a BIGINT, or the integer type `planned` names when `v` fits it.
    DUCKDB_V2_ERROR convert_int(int64_t v, DUCKDB_V2_LOGICAL_TYPE_ID planned,
                                duckdb_v2_value_handle* out, duckdb_v2_error_info_handle* err);

    duckdb_v2_factory_handle factory_;
    std::vector<std::string> names_;
    std::vector<const char*> name_ptrs_;
    std::vector<idx_t> name_lengths_;
    std::vector<duckdb_v2_value_handle> values_;
};

V2Parameters::V2Parameters(duckdb_v2_factory_handle factory, nb::handle parameters,
                           const V2Prepared* planned)
    : factory_(factory) {
    if (parameters.is_none()) return;
    auto planned_type = [&](const std::string& key) {
        return planned ? planned->planned_type(key) : DUCKDB_V2_LOGICAL_TYPE_ID_INVALID;
    };
    // A throwing constructor runs no destructor.
    try {
        idx_t idx = 0;
        if (nb::isinstance<nb::dict>(parameters)) {
            for (auto [k, v] : nb::borrow<nb::dict>(parameters)) {
                names_.push_back(nb::cast<std::string>(k));
                if (planned && planned->parameter_index(names_.back()) < 0) {
                    throw DuckyError("ducky: unknown named parameter '" + names_.back() + "'");
                }
                values_.push_back(convert(v, ++idx, planned_type(names_.back())));
            }
            for (const std::string& name : names_) {
                name_ptrs_.push_back(name.data());
                name_lengths_.push_back(name.size());
            }
        } else {
            for (nb::handle v : parameters) {
                ++idx;
                values_.push_back(convert(v, idx, planned_type(std::to_string(idx))));
            }
        }
    } catch (...) {
        for (duckdb_v2_value_handle& value : values_) duckdb_v2_value_destroy(&value);
        throw;
    }
}

DUCKDB_V2_ERROR V2Parameters::convert_int(int64_t v, DUCKDB_V2_LOGICAL_TYPE_ID planned,
                                          duckdb_v2_value_handle* out,
                                          duckdb_v2_error_info_handle* err) {
    switch (planned) {
        case DUCKDB_V2_LOGICAL_TYPE_ID_TINYINT:
            if (v >= INT8_MIN && v <= INT8_MAX) {
                return duckdb_v2_value_create_tinyint(factory_, (int8_t)v, out, err);
            }
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_SMALLINT:
            if (v >= INT16_MIN && v <= INT16_MAX) {
                return duckdb_v2_value_create_smallint(factory_, (int16_t)v, out, err);
            }
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_INTEGER:
            if (v >= INT32_MIN && v <= INT32_MAX) {
                return duckdb_v2_value_create_int(factory_, (int32_t)v, out, err);
            }
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_UTINYINT:
            if (v >= 0 && v <= UINT8_MAX) {
                return duckdb_v2_value_create_utinyint(factory_, (uint8_t)v, out, err);
            }
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_USMALLINT:
            if (v >= 0 && v <= UINT16_MAX) {
                return duckdb_v2_value_create_usmallint(factory_, (uint16_t)v, out, err);
            }
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_UINTEGER:
            if (v >= 0 && v <= (int64_t)UINT32_MAX) {
                return duckdb_v2_value_create_uint(factory_, (uint32_t)v, out, err);
            }
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_UBIGINT:
            if (v >= 0) return duckdb_v2_value_create_ubigint(factory_, (uint64_t)v, out, err);
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_HUGEINT: {
            // Any int64 fits; sign-extend into the upper half.
            duckdb_v2_hugeint_t h{(uint64_t)v, v < 0 ? (int64_t)-1 : (int64_t)0};
            return duckdb_v2_value_create_hugeint(factory_, &h, out, err);
        }
        default:
            // BIGINT, an unresolved type, and everything a BIGINT must be cast
            // into (DECIMAL, DOUBLE, ...).
            break;
    }
    return duckdb_v2_value_create_bigint(factory_, v, out, err);
}

duckdb_v2_value_handle V2Parameters::convert(nb::handle value, idx_t idx,
                                             DUCKDB_V2_LOGICAL_TYPE_ID planned) {
    const BindTypes& T = bind_types();
    duckdb_v2_value_handle out = nullptr;
    V2Error err;
    auto made = [&](DUCKDB_V2_ERROR code) {
        err.check(code, "ducky: failed to bind parameter " + std::to_string(idx) + ": ");
        return out;
    };

    if (value.is_none()) {
        // An untyped NULL, as duckdb_bind_null binds it; no type id creates
        // SQLNULL.
        duckdb_v2_logical_type_handle null_type = nullptr;
        made(duckdb_v2_factory_create_type_from_text(factory_, "NULL", 4, &null_type, err.out()));
        DUCKDB_V2_ERROR code = duckdb_v2_value_create_null(factory_, null_type, &out, err.out());
        duckdb_v2_logical_type_destroy(&null_type);
        return made(code);
    }
    if (nb::isinstance<nb::bool_>(value)) {
        return made(duckdb_v2_value_create_bool(factory_, nb::cast<bool>(value), &out, err.out()));
    }
    if (nb::isinstance<nb::int_>(value)) {
        int64_t v = PyLong_AsLongLong(value.ptr());
        if (v != -1 || !PyErr_Occurred()) {
            return made(convert_int(v, planned, &out, err.out()));
        }
        PyErr_Clear();
        bool negative = PyObject_RichCompareBool(value.ptr(), nb::int_(0).ptr(), Py_LT) == 1;
        uint64_t lo = 0, hi = 0;
        py_long_to_int128(value, /*is_signed=*/negative, lo, hi);
        if (negative) {
            duckdb_v2_hugeint_t h{lo, (int64_t)hi};
            return made(duckdb_v2_value_create_hugeint(factory_, &h, &out, err.out()));
        }
        duckdb_v2_uhugeint_t h{lo, hi};
        return made(duckdb_v2_value_create_uhugeint(factory_, &h, &out, err.out()));
    }
    if (nb::isinstance<nb::float_>(value)) {
        double v = nb::cast<double>(value);
        if (planned == DUCKDB_V2_LOGICAL_TYPE_ID_FLOAT) {
            return made(duckdb_v2_value_create_float(factory_, (float)v, &out, err.out()));
        }
        return made(duckdb_v2_value_create_double(factory_, v, &out, err.out()));
    }
    if (nb::isinstance<nb::str>(value)) {
        Py_ssize_t size = 0;
        const char* data = PyUnicode_AsUTF8AndSize(value.ptr(), &size);
        if (!data) throw nb::python_error();
        return made(duckdb_v2_value_create_varchar(factory_, data, (idx_t)size, &out, err.out()));
    }
    if (nb::isinstance<nb::bytes>(value)) {
        char* data = nullptr;
        Py_ssize_t size = 0;
        if (PyBytes_AsStringAndSize(value.ptr(), &data, &size) != 0) throw nb::python_error();
        return made(duckdb_v2_value_create_blob(factory_, data, (idx_t)size, &out, err.out()));
    }
    // datetime.datetime is a subclass of datetime.date, so check it first.
    PyObject* raw = value.ptr();
    if (PyDateTime_Check(raw)) {
        if (!value.attr("tzinfo").is_none()) {
            return made(duckdb_v2_value_create_timestamp_tz(
                factory_, aware_datetime_to_utc_micros(value), &out, err.out()));
        }
        duckdb_timestamp_struct s{};
        s.date.year = PyDateTime_GET_YEAR(raw);
        s.date.month = (int8_t)PyDateTime_GET_MONTH(raw);
        s.date.day = (int8_t)PyDateTime_GET_DAY(raw);
        s.time.hour = (int8_t)PyDateTime_DATE_GET_HOUR(raw);
        s.time.min = (int8_t)PyDateTime_DATE_GET_MINUTE(raw);
        s.time.sec = (int8_t)PyDateTime_DATE_GET_SECOND(raw);
        s.time.micros = PyDateTime_DATE_GET_MICROSECOND(raw);
        return made(duckdb_v2_value_create_timestamp(factory_, duckdb_to_timestamp(s).micros, &out,
                                                     err.out()));
    }
    if (PyDate_Check(raw)) {
        duckdb_date_struct s{};
        s.year = PyDateTime_GET_YEAR(raw);
        s.month = (int8_t)PyDateTime_GET_MONTH(raw);
        s.day = (int8_t)PyDateTime_GET_DAY(raw);
        return made(duckdb_v2_value_create_date(factory_, duckdb_to_date(s).days, &out, err.out()));
    }
    if (PyTime_Check(raw)) {
        duckdb_time_struct s{};
        s.hour = (int8_t)PyDateTime_TIME_GET_HOUR(raw);
        s.min = (int8_t)PyDateTime_TIME_GET_MINUTE(raw);
        s.sec = (int8_t)PyDateTime_TIME_GET_SECOND(raw);
        s.micros = PyDateTime_TIME_GET_MICROSECOND(raw);
        return made(
            duckdb_v2_value_create_time(factory_, duckdb_to_time(s).micros, &out, err.out()));
    }
    if (PyDelta_Check(raw)) {
        int64_t seconds = PyDateTime_DELTA_GET_SECONDS(raw);
        duckdb_v2_interval_t iv{0, PyDateTime_DELTA_GET_DAYS(raw),
                                seconds * 1000000 + PyDateTime_DELTA_GET_MICROSECONDS(raw)};
        return made(duckdb_v2_value_create_interval(factory_, &iv, &out, err.out()));
    }
    if (nb::isinstance(value, T.decimal)) {
        std::optional<duckdb_decimal> d = decimal_parts(value, idx);
        if (!d) {
            return made(duckdb_v2_value_create_double(factory_, nb::cast<double>(nb::float_(value)),
                                                      &out, err.out()));
        }
        duckdb_v2_hugeint_t h{d->value.lower, d->value.upper};
        return made(
            duckdb_v2_value_create_decimal(factory_, &h, d->width, d->scale, &out, err.out()));
    }
    if (nb::isinstance(value, T.uuid)) {
        // UUID.int is the natural unsigned value; storage flips its top bit so
        // UUIDs sort as signed integers.
        uint64_t lo = 0, hi = 0;
        py_long_to_int128(value.attr("int"), /*is_signed=*/false, lo, hi);
        duckdb_v2_hugeint_t h{lo, (int64_t)(hi ^ (uint64_t(1) << 63))};
        return made(duckdb_v2_value_create_uuid(factory_, &h, &out, err.out()));
    }
    // Unbox NumPy scalars.
    if (nb::hasattr(value, "item")) {
        nb::object item;
        try {
            item = value.attr("item")();
        } catch (...) {
            item = nb::object();
        }
        if (item && item.type().ptr() != value.type().ptr()) return convert(item, idx, planned);
    }

    throw DuckyError(std::string("ducky: unsupported parameter type at position ") +
                     std::to_string(idx) + " (" + nb::cast<std::string>(nb::str(value.type())) +
                     ")");
}

void arrow_stream_capsule_destructor(PyObject* capsule) {
    auto* stream = (ArrowArrayStream*)PyCapsule_GetPointer(capsule, "arrow_array_stream");
    if (stream) {
        if (stream->release) stream->release(stream);
        free(stream);
    }
}

// The arrays of a finished v2 Arrow result, served as an ArrowArrayStream.
// The result stays open to report the schema, and the handle to keep the
// result's connection.
struct CollectedArrow {
    std::shared_ptr<DuckDBHandle> handle;
    duckdb_v2_arrow_result_handle result = nullptr;
    std::vector<ArrowArray> arrays;
    size_t next = 0;
    std::string error;

    ~CollectedArrow() {
        release_arrays(arrays, next);
        duckdb_v2_arrow_result_destroy(&result);
    }

    static void release_arrays(std::vector<ArrowArray>& arrays, size_t from = 0) {
        for (size_t i = from; i < arrays.size(); i++) {
            if (arrays[i].release) arrays[i].release(&arrays[i]);
        }
    }

    static CollectedArrow& of(ArrowArrayStream* stream) {
        return *(CollectedArrow*)stream->private_data;
    }
    static int get_schema(ArrowArrayStream* stream, ArrowSchema* out) {
        CollectedArrow& self = of(stream);
        V2Error err;
        if (duckdb_v2_arrow_result_get_schema(self.result, out, err.out()) !=
            DUCKDB_V2_ERROR_NONE) {
            self.error = err.text();
            return EIO;
        }
        return 0;
    }
    static int get_next(ArrowArrayStream* stream, ArrowArray* out) {
        CollectedArrow& self = of(stream);
        if (self.next == self.arrays.size()) {
            out->release = nullptr;
            return 0;
        }
        *out = self.arrays[self.next];
        self.arrays[self.next++].release = nullptr;
        return 0;
    }
    static const char* get_last_error(ArrowArrayStream* stream) {
        CollectedArrow& self = of(stream);
        return self.error.empty() ? nullptr : self.error.c_str();
    }
    static void release(ArrowArrayStream* stream) {
        delete &of(stream);
        stream->release = nullptr;
    }
};

// Steps `result` to its end with the GIL released and returns its arrays.
// Resurfaces every 10 ms for Python signals; one interrupts the query.
std::vector<ArrowArray> collect_arrow(duckdb_v2_connection_handle con,
                                      duckdb_v2_arrow_result_handle result) {
    std::vector<ArrowArray> arrays;
    try {
        V2Error err;
        for (;;) {
            DUCKDB_V2_ERROR code;
            DUCKDB_V2_RESULT_STEP_STATUS status;
            {
                nb::gil_scoped_release release;
                auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
                do {
                    ArrowArray array{};
                    code = duckdb_v2_arrow_result_step(result, &array, &status, err.out());
                    if (code != DUCKDB_V2_ERROR_NONE) break;
                    if (status == DUCKDB_V2_RESULT_STEP_STATUS_CHUNK) {
                        arrays.push_back(array);
                    } else if (status == DUCKDB_V2_RESULT_STEP_STATUS_WAITING) {
                        // Bounded, and woken when a task completes.
                        duckdb_v2_arrow_result_wait(result, nullptr);
                    } else {
                        break;
                    }
                } while (std::chrono::steady_clock::now() < deadline);
            }
            err.check(code);
            if (status == DUCKDB_V2_RESULT_STEP_STATUS_FINISHED) return arrays;
            if (status == DUCKDB_V2_RESULT_STEP_STATUS_CANCELLED) {
                throw DuckyError(kInterrupted);
            }
            if (PyErr_CheckSignals() != 0) {
                // Typically KeyboardInterrupt. The signal's exception stays set
                // while the query winds down; its workers may need the GIL.
                nb::gil_scoped_release release;
                duckdb_v2_connection_interrupt(con, nullptr);
                for (;;) {
                    ArrowArray array{};
                    if (duckdb_v2_arrow_result_step(result, &array, &status, nullptr) !=
                            DUCKDB_V2_ERROR_NONE ||
                        status == DUCKDB_V2_RESULT_STEP_STATUS_FINISHED ||
                        status == DUCKDB_V2_RESULT_STEP_STATUS_CANCELLED) {
                        break;
                    }
                    if (array.release) array.release(&array);
                    duckdb_v2_arrow_result_wait(result, nullptr);
                }
                throw nb::python_error();
            }
        }
    } catch (...) {
        CollectedArrow::release_arrays(arrays);
        throw;
    }
}

// Owns the single statement of `query`, parsed for execution.
class ParsedStatement {
   public:
    ParsedStatement(duckdb_v2_connection_handle con, const std::string& query) {
        V2Error err;
        duckdb_v2_statement_iterator_handle statements = nullptr;
        err.check(duckdb_v2_parse_sql(con, query.data(), query.size(),
                                      /*apply_invalidation_policy=*/true, &statements, err.out()));
        duckdb_v2_sql_statement_handle extra = nullptr;
        DUCKDB_V2_ERROR code = duckdb_v2_statement_iterator_next(statements, &stmt_, err.out());
        if (code == DUCKDB_V2_ERROR_NONE && stmt_) {
            code = duckdb_v2_statement_iterator_next(statements, &extra, err.out());
        }
        duckdb_v2_statement_iterator_destroy(&statements);
        if (extra) {
            duckdb_v2_sql_statement_destroy(&extra);
            duckdb_v2_sql_statement_destroy(&stmt_);
            throw DuckyError("ducky: expected one statement, got several");
        }
        if (code != DUCKDB_V2_ERROR_NONE) {
            duckdb_v2_sql_statement_destroy(&stmt_);
            err.check(code);
        }
        if (!stmt_) throw DuckyError("ducky: no statement to execute");
    }
    ~ParsedStatement() { duckdb_v2_sql_statement_destroy(&stmt_); }

    ParsedStatement(const ParsedStatement&) = delete;
    ParsedStatement& operator=(const ParsedStatement&) = delete;

    duckdb_v2_sql_statement_handle get() const { return stmt_; }
    // Hands the statement over to the caller.
    duckdb_v2_sql_statement_handle release() { return std::exchange(stmt_, nullptr); }

   private:
    duckdb_v2_sql_statement_handle stmt_ = nullptr;
};

// Runs `execution` until it settles, with the GIL released, resurfacing every
// 10 ms for Python signals. A signal interrupts the query and raises; the query
// ends as the execution is destroyed.
void drive(duckdb_v2_connection_handle con, Execution& execution) {
    for (;;) {
        bool settled;
        {
            nb::gil_scoped_release release;
            settled =
                execution.run(std::chrono::steady_clock::now() + std::chrono::milliseconds(10));
        }
        if (settled) return;
        if (PyErr_CheckSignals() != 0) {
            // Typically KeyboardInterrupt.
            duckdb_v2_connection_interrupt(con, nullptr);
            throw nb::python_error();
        }
    }
}

// Runs `result` to its end with the GIL released, discarding its rows, and
// destroys it.
DUCKDB_V2_ERROR drain(duckdb_v2_result_handle& result, duckdb_v2_error_info_handle* err) {
    idx_t rows_changed = 0;
    DUCKDB_V2_ERROR code;
    {
        // Execution may run Python UDFs on DuckDB workers.
        release_gil_if_held release;
        code = duckdb_v2_result_drain(result, &rows_changed, err);
        duckdb_v2_result_destroy(&result);
    }
    return code;
}

// Runs `sql`, one statement without parameters, to its end. Raises DuckDB's
// message after `prefix` on failure.
void run_statement(duckdb_v2_connection_handle con, const std::string& sql,
                   const std::string& prefix) {
    ParsedStatement stmt(con, sql);
    V2Error err;
    duckdb_v2_result_handle result = nullptr;
    DUCKDB_V2_ERROR code;
    {
        // Takes the client context lock, which a query running Python
        // callbacks on another thread may hold while it waits for the GIL.
        release_gil_if_held release;
        code = duckdb_v2_statement_execute(con, stmt.get(), nullptr, nullptr, nullptr, 0, &result,
                                           err.out());
    }
    if (code == DUCKDB_V2_ERROR_NONE) code = drain(result, err.out());
    err.check(code, prefix);
}

duckdb_v2_factory_handle connection_factory(duckdb_v2_connection_handle con) {
    V2Error err;
    duckdb_v2_factory_handle factory = nullptr;
    err.check(duckdb_v2_connection_get_factory(con, &factory, err.out()));
    return factory;
}

// Executes `stmt` with `parameters`, returning its unstepped result.
duckdb_v2_result_handle execute_prepared(duckdb_v2_connection_handle con, const V2Prepared& stmt,
                                         nb::handle parameters) {
    V2Parameters values(connection_factory(con), parameters, &stmt);
    V2Error err;
    duckdb_v2_result_handle result = nullptr;
    DUCKDB_V2_ERROR code;
    {
        release_gil_if_held release;
        code = duckdb_v2_prepared_statement_execute(stmt.get(), values.names(),
                                                    values.name_lengths(), values.values(),
                                                    values.count(), &result, err.out());
    }
    err.check(code);
    return result;
}

// Runs `stmt` once per parameter set in `seq`, discarding the results, with
// periodic signal checks.
void executemany_loop(duckdb_v2_connection_handle con, const V2Prepared& stmt, nb::handle seq) {
    constexpr idx_t SIGNAL_CHECK_PERIOD = 1024;
    idx_t since_check = 0;
    V2Error err;
    for (nb::handle params : seq) {
        // Each execution binds only its own values, so a short positional row
        // or a named dict missing a key fails instead of inheriting the
        // previous row's.
        duckdb_v2_result_handle result = execute_prepared(con, stmt, params);
        err.check(drain(result, err.out()));
        if (++since_check >= SIGNAL_CHECK_PERIOD) {
            since_check = 0;
            if (PyErr_CheckSignals() != 0) throw nb::python_error();
        }
    }
}

}  // namespace

Connection::Connection(const std::string& database, nb::object config) {
    auto handle = std::make_shared<DuckDBHandle>();

    // Iterate via .items() so we accept any read-only Mapping (TypedDict,
    // MappingProxy, user mappings), not just nb::dict.
    std::vector<std::string> names, values;
    if (!config.is_none()) {
        for (nb::handle item : config.attr("items")()) {
            nb::tuple kv = nb::cast<nb::tuple>(item);
            names.push_back(nb::cast<std::string>(kv[0]));
            values.push_back(nb::cast<std::string>(kv[1]));
        }
    }
    // An instance has no database path to put its spill files next to, as
    // opening one does.
    std::string path = database.empty() ? ":memory:" : database;
    if (path != ":memory:" &&
        std::find(names.begin(), names.end(), "temp_directory") == names.end()) {
        names.push_back("temp_directory");
        values.push_back(path + ".tmp");
    }
    std::vector<const char*> name_ptrs, value_ptrs;
    std::vector<idx_t> name_lengths, value_lengths;
    for (size_t i = 0; i < names.size(); i++) {
        name_ptrs.push_back(names[i].data());
        name_lengths.push_back(names[i].size());
        value_ptrs.push_back(values[i].data());
        value_lengths.push_back(values[i].size());
    }

    // An instance starts with no database: attach the path as its default, the
    // way opening it would. ~DuckDBHandle tears down what exists as `handle`
    // unwinds.
    std::string failed = "ducky: failed to open '" + database + "': ";
    V2Error err;
    err.check(duckdb_v2_instance_create_with_options(
                  v2_environment(), name_ptrs.data(), name_lengths.data(), value_ptrs.data(),
                  value_lengths.data(), names.size(), &handle->instance, err.out()),
              failed);
    err.check(duckdb_v2_instance_attach(handle->instance, "", 0, path.data(), path.size(), nullptr,
                                        /*make_default=*/true, err.out()),
              failed);
    err.check(duckdb_v2_connection_create(handle->instance, &handle->connection_v2, err.out()),
              "ducky: failed to connect to '" + database + "': ");
    handle->connection = duckdb_v2_connection_get_v1(handle->connection_v2);
    handle_ = std::move(handle);
    profile_sink_ = nb::none();
}

std::shared_ptr<DuckDBHandle> Connection::handle() const {
    nb::ft_lock_guard lock(handle_mu_);
    return handle_;
}

std::shared_ptr<DuckDBHandle> Connection::open_handle() const {
    std::shared_ptr<DuckDBHandle> handle = this->handle();
    if (!handle || !handle->connection) throw DuckyError("ducky: connection is closed");
    return handle;
}

void Connection::interrupt() {
    // The snapshot keeps the connection alive across the call.
    if (std::shared_ptr<DuckDBHandle> handle = this->handle()) {
        if (handle->connection_v2) duckdb_v2_connection_interrupt(handle->connection_v2, nullptr);
    }
}

void Connection::load_extension(const std::string& name) {
    std::shared_ptr<DuckDBHandle> handle = open_handle();

    // Build `LOAD '<name>'`, doubling single quotes for the SQL string literal.
    std::string literal;
    literal.reserve(name.size() + 2);
    for (char c : name) {
        if (c == '\'') literal.push_back('\'');
        literal.push_back(c);
    }
    // dlopen + metadata/signature checks touch the filesystem; run_statement
    // releases the GIL so other threads run during the load.
    run_statement(handle->connection_v2, "LOAD '" + literal + "'",
                  "ducky: failed to load extension '" + name + "': ");

    // LOAD can register new functions / types / settings, so drop cached plans
    // — mirrors the caches_statement() flush run() applies to LOAD statements.
    statement_cache_clear();
}

nb::tuple Connection::progress() const {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    double percentage = -1;
    uint64_t rows_processed = 0, total_rows_to_process = 0;
    V2Error err;
    err.check(duckdb_v2_connection_progress_get(handle->connection_v2, &percentage, &rows_processed,
                                                &total_rows_to_process, err.out()));
    return nb::make_tuple(percentage, rows_processed, total_rows_to_process);
}

namespace {

// Owns a query metrics snapshot.
struct V2Metrics {
    duckdb_v2_query_metrics_handle handle = nullptr;
    V2Metrics() = default;
    V2Metrics(const V2Metrics&) = delete;
    V2Metrics& operator=(const V2Metrics&) = delete;
    ~V2Metrics() { duckdb_v2_query_metrics_destroy(&handle); }
};

// A node's metrics as name and text, the text as v1 rendered it.
std::vector<std::pair<std::string, std::string>> read_metrics(
    duckdb_v2_query_metrics_node_handle node) {
    std::vector<std::pair<std::string, std::string>> out;
    idx_t count = 0;
    duckdb_v2_query_metrics_node_get_metric_count(node, &count, nullptr);
    std::vector<char> text;
    for (idx_t i = 0; i < count; ++i) {
        const char* name = nullptr;
        idx_t name_length = 0;
        duckdb_v2_value_handle value = nullptr;
        if (duckdb_v2_query_metrics_node_get_metric_name(node, i, &name, &name_length, nullptr) !=
                DUCKDB_V2_ERROR_NONE ||
            duckdb_v2_query_metrics_node_get_metric_value(node, i, &value, nullptr) !=
                DUCKDB_V2_ERROR_NONE) {
            continue;
        }
        idx_t length = 0;
        if (duckdb_v2_value_to_string(value, nullptr, 0, &length, nullptr) ==
            DUCKDB_V2_ERROR_NONE) {
            text.resize(length + 1);
            if (duckdb_v2_value_to_string(value, text.data(), text.size(), &length, nullptr) ==
                DUCKDB_V2_ERROR_NONE) {
                out.emplace_back(std::string(name, name_length), std::string(text.data(), length));
            }
        }
        duckdb_v2_value_destroy(&value);
    }
    return out;
}

nb::dict profile_node(nb::dict metrics, nb::list children) {
    nb::dict node;
    node["metrics"] = std::move(metrics);
    node["children"] = std::move(children);
    return node;
}

// An operator node and its children, with metrics keyed without `operator.`.
nb::dict walk_operator(duckdb_v2_query_metrics_node_handle node) {
    constexpr std::string_view prefix = "operator.";
    nb::dict metrics;
    for (auto& [name, text] : read_metrics(node)) {
        std::string_view key = name;
        if (key.substr(0, prefix.size()) == prefix) key.remove_prefix(prefix.size());
        metrics[nb::str(key.data(), key.size())] = nb::str(text.data(), text.size());
    }
    nb::list children;
    idx_t count = 0;
    duckdb_v2_query_metrics_node_get_child_count(node, &count, nullptr);
    for (idx_t i = 0; i < count; ++i) {
        duckdb_v2_query_metrics_node_handle child = nullptr;
        if (duckdb_v2_query_metrics_node_get_child(node, i, &child, nullptr) ==
            DUCKDB_V2_ERROR_NONE) {
            children.append(walk_operator(child));
        }
    }
    return profile_node(std::move(metrics), std::move(children));
}

// The last query's profile in the shape v1's profiling info had: an empty root
// whose children are the operator tree and one node per metric group (query,
// system, io, planner, ...), keyed without the group. None without a plan.
nb::object profile_tree(duckdb_v2_connection_handle con) {
    V2Metrics metrics;
    V2Error err;
    DUCKDB_V2_ERROR code;
    {
        release_gil_if_held release;
        code = duckdb_v2_connection_get_query_metrics(con, &metrics.handle, err.out());
    }
    // v1 reported the previous query's profile while a result streamed.
    if (code == DUCKDB_V2_ERROR_RESOURCE_IN_USE) {
        throw DuckyError("ducky: no profile while the connection has an open result");
    }
    err.check(code, "ducky: ");
    duckdb_v2_query_metrics_node_handle root = nullptr;
    err.check(duckdb_v2_query_metrics_get_root(metrics.handle, &root, err.out()), "ducky: ");
    idx_t count = 0;
    duckdb_v2_query_metrics_node_get_child_count(root, &count, nullptr);
    if (count == 0) return nb::none();

    nb::list children;
    for (idx_t i = 0; i < count; ++i) {
        duckdb_v2_query_metrics_node_handle child = nullptr;
        if (duckdb_v2_query_metrics_node_get_child(root, i, &child, nullptr) ==
            DUCKDB_V2_ERROR_NONE) {
            children.append(walk_operator(child));
        }
    }
    std::vector<std::pair<std::string, nb::dict>> groups;
    for (auto& [name, text] : read_metrics(root)) {
        size_t dot = name.find('.');
        std::string group = dot == std::string::npos ? std::string() : name.substr(0, dot);
        std::string key = dot == std::string::npos ? name : name.substr(dot + 1);
        auto it = std::find_if(groups.begin(), groups.end(),
                               [&](const auto& entry) { return entry.first == group; });
        if (it == groups.end()) it = groups.insert(groups.end(), {group, nb::dict()});
        it->second[nb::str(key.data(), key.size())] = nb::str(text.data(), text.size());
    }
    for (auto& [group, group_metrics] : groups) {
        children.append(profile_node(std::move(group_metrics), nb::list()));
    }
    return profile_node(nb::dict(), std::move(children));
}

}  // namespace

nb::object Connection::get_profiling_info() const {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    return profile_tree(handle->connection_v2);
}

namespace {

// Run a SET (or other no-result) statement, bypassing run() so it doesn't
// reach the profile sink.
void exec_no_result(duckdb_v2_connection_handle con, const std::string& sql) {
    run_statement(con, sql, "ducky: " + sql + " failed: ");
}

}  // namespace

void Connection::set_profile_sink(nb::object sink, int64_t sample, const std::string& mode) {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    if (sample < 1) sample = 1;
    if (mode != "standard" && mode != "detailed") {
        throw DuckyError("ducky: profile mode must be 'standard' or 'detailed', got '" + mode +
                         "'");
    }
    if (sink.is_none()) {
        profile_sink_ = nb::none();
        profile_sample_ = 1;
        profile_counter_ = 0;
        return;
    }
    if (!PyCallable_Check(sink.ptr())) {
        throw DuckyError("ducky: profile sink must be callable or None");
    }
    duckdb_v2_connection_handle con = handle->connection_v2;
    exec_no_result(con, "SET enable_profiling='no_output'");
    // profiling_mode persists, so always set it.
    exec_no_result(con, "SET profiling_mode='" + mode + "'");
    profile_sink_ = sink;
    profile_sample_ = sample;
    profile_counter_ = 0;
}

void Connection::maybe_emit_profile(duckdb_v2_connection_handle connection,
                                    const std::string& query) {
    // Fast path: no sink installed.
    if (profile_sink_.is_none()) return;
    int64_t n = profile_counter_++;
    if (n % profile_sample_ != 0) return;
    nb::object tree = profile_tree(connection);
    if (tree.is_none()) return;
    // Report sink errors without failing the query.
    try {
        profile_sink_(query, tree);
    } catch (nb::python_error& e) {
        e.restore();
        PyErr_WriteUnraisable(profile_sink_.ptr());
    }
}

void Connection::register_arrow(const std::string& name, nb::object obj) {
    open_handle();
    // Keep a strict identifier shape so `SELECT * FROM name` resolves unquoted:
    // [A-Za-z][A-Za-z0-9_]*.
    if (name.empty()) {
        throw DuckyError("ducky: empty table name");
    }
    auto ident_ok = [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
               c == '_';
    };
    if (!((name[0] >= 'a' && name[0] <= 'z') || (name[0] >= 'A' && name[0] <= 'Z'))) {
        throw DuckyError("ducky: invalid table name '" + name + "' (must start with a letter)");
    }
    for (char c : name) {
        if (!ident_ok(c)) {
            throw DuckyError("ducky: invalid table name '" + name + "' (allowed: [A-Za-z0-9_])");
        }
    }
    if (!nb::hasattr(obj, "__arrow_c_stream__")) {
        throw DuckyError(
            "ducky: object does not implement the Arrow PyCapsule interface "
            "(__arrow_c_stream__)");
    }

    if (!arrow_registry_) arrow_registry_ = install_arrow_scan(*this);
    // Re-registering may change the schema.
    statement_cache_clear();
    register_arrow_source(*arrow_registry_, name, std::move(obj));
    open_handle()->arrow_generation++;
}

Connection::~Connection() { close(); }

void Connection::close() {
    std::shared_ptr<DuckDBHandle> handle;
    {
        nb::ft_lock_guard lock(handle_mu_);
        handle = std::move(handle_);
    }
    // Stop a query another thread is running; it holds its own reference to
    // the handle, so the connection stays valid until that call returns.
    // DuckDB clears the interrupt flag when the next query starts, so results
    // and prepared statements that outlive the Connection are unaffected.
    if (handle && handle->connection_v2) {
        duckdb_v2_connection_interrupt(handle->connection_v2, nullptr);
    }
    statement_cache_clear();
    last_result_.reset();
    if (arrow_registry_) {
        std::lock_guard<std::mutex> lock(arrow_registry_->mu);
        arrow_registry_->sources.clear();
    }
    profile_sink_ = nb::none();
    // Results, prepared statements and appenders retain shared ownership of
    // the database; the last owner closes it.
}

int Connection::gc_traverse(visitproc visit, void* arg) const {
    Py_VISIT(profile_sink_.ptr());
    // The database owns its functions' callables. Only a sole owner of the
    // database owns them, and only then can gc_clear free them by closing.
    if (handle_ && handle_.use_count() == 1) {
        for (const nb::object& obj : handle_->py_refs()) Py_VISIT(obj.ptr());
    }
    // GC holds the GIL or stops all threads; taking mu could deadlock.
    if (arrow_registry_) {
        for (const auto& [name, source] : arrow_registry_->sources) {
            Py_VISIT(source.object.ptr());
            Py_VISIT(source.stream.ptr());
        }
    }
    return 0;
}

void Connection::gc_clear() {
    if (handle_ && handle_.use_count() == 1) {
        // Closing drops the database, and with it the callables it owns.
        close();
        return;
    }
    if (arrow_registry_) {
        std::lock_guard<std::mutex> lock(arrow_registry_->mu);
        arrow_registry_->sources.clear();
    }
    profile_sink_ = nb::none();
}

namespace {

constexpr size_t kStatementCacheCapacity = 128;

// Cache queries whose plans remain valid across repeated use.
bool caches_statement(DUCKDB_V2_STATEMENT_TYPE type) {
    switch (type) {
        case DUCKDB_V2_STATEMENT_TYPE_SELECT:
        case DUCKDB_V2_STATEMENT_TYPE_INSERT:
        case DUCKDB_V2_STATEMENT_TYPE_UPDATE:
        case DUCKDB_V2_STATEMENT_TYPE_DELETE:
        case DUCKDB_V2_STATEMENT_TYPE_EXPLAIN:
        case DUCKDB_V2_STATEMENT_TYPE_COPY:
        case DUCKDB_V2_STATEMENT_TYPE_ANALYZE:
        case DUCKDB_V2_STATEMENT_TYPE_TRANSACTION:
        case DUCKDB_V2_STATEMENT_TYPE_RELATION:
            return true;
        default:
            return false;
    }
}

}  // namespace

V2Prepared::V2Prepared(duckdb_v2_connection_handle con, const std::string& query) {
    ParsedStatement parsed(con, query);
    V2Error err;
    err.check(duckdb_v2_sql_statement_get_type(parsed.get(), &type_, err.out()));
    idx_t count = 0;
    err.check(duckdb_v2_sql_statement_get_parameter_count(parsed.get(), &count, err.out()));
    for (idx_t i = 0; i < count; i++) {
        const char* name = nullptr;
        idx_t length = 0;
        err.check(
            duckdb_v2_sql_statement_get_parameter_name(parsed.get(), i, &name, &length, err.out()));
        parameter_names_.emplace_back(name, length);
    }
    DUCKDB_V2_ERROR code;
    if (count > 0) {
        // The parameter types the plan will assume, so executions can pass
        // values of those types and reuse it. A statement that expands into
        // several has none, and takes no parameters anyway.
        duckdb_v2_schema_handle columns = nullptr, params = nullptr;
        {
            // Binding takes the client context lock.
            release_gil_if_held release;
            code = duckdb_v2_statement_bind(con, parsed.get(), &columns, &params, err.out());
        }
        duckdb_v2_schema_destroy(&columns);
        idx_t n = 0;
        if (code == DUCKDB_V2_ERROR_NONE &&
            duckdb_v2_schema_get_count(params, &n, nullptr) == DUCKDB_V2_ERROR_NONE) {
            for (idx_t i = 0; i < n; i++) {
                const char* name = nullptr;
                idx_t length = 0;
                duckdb_v2_logical_type_handle type = nullptr;
                DUCKDB_V2_LOGICAL_TYPE_ID id = DUCKDB_V2_LOGICAL_TYPE_ID_INVALID;
                if (duckdb_v2_schema_get_field(params, i, &name, &length, &type, nullptr) ==
                    DUCKDB_V2_ERROR_NONE) {
                    duckdb_v2_logical_type_get_id(type, &id, nullptr);
                }
                parameter_types_.push_back(id);
            }
        }
        duckdb_v2_schema_destroy(&params);
    }
    {
        // Preparing takes the client context lock.
        release_gil_if_held release;
        code = duckdb_v2_prepared_statement_create(con, parsed.get(), /*require_cacheable=*/false,
                                                   &prepared_, err.out());
    }
    err.check(code);
    // Kept for binding the statement's result schema.
    sql_ = parsed.release();
}

V2Prepared::~V2Prepared() {
    // Destroying the plan takes the client context lock.
    release_gil_if_held release;
    duckdb_v2_prepared_statement_destroy(&prepared_);
    duckdb_v2_sql_statement_destroy(&sql_);
}

int64_t V2Prepared::parameter_index(const std::string& key) const {
    // Named parameters match case-insensitively, as DuckDB binds them.
    auto lower = [](std::string text) {
        for (char& c : text) c = (char)std::tolower((unsigned char)c);
        return text;
    };
    std::string wanted = lower(key);
    for (size_t i = 0; i < parameter_names_.size(); i++) {
        if (lower(parameter_names_[i]) == wanted) return (int64_t)i;
    }
    return -1;
}

DUCKDB_V2_LOGICAL_TYPE_ID V2Prepared::planned_type(const std::string& key) const {
    int64_t i = parameter_index(key);
    if (i < 0 || (size_t)i >= parameter_types_.size()) return DUCKDB_V2_LOGICAL_TYPE_ID_INVALID;
    return parameter_types_[(size_t)i];
}

std::vector<std::pair<std::string, std::string>> V2Prepared::columns(
    duckdb_v2_connection_handle con) const {
    V2Error err;
    duckdb_v2_schema_handle schema = nullptr;
    DUCKDB_V2_ERROR code;
    {
        release_gil_if_held release;
        code = duckdb_v2_statement_bind(con, sql_, &schema, nullptr, err.out());
    }
    std::vector<std::pair<std::string, std::string>> out;
    if (code == DUCKDB_V2_ERROR_INPUT_METADATA_DEFERRED) return out;
    err.check(code);
    idx_t n = 0;
    code = duckdb_v2_schema_get_count(schema, &n, err.out());
    for (idx_t i = 0; i < n && code == DUCKDB_V2_ERROR_NONE; i++) {
        const char* name = nullptr;
        idx_t length = 0;
        duckdb_v2_logical_type_handle type = nullptr;
        code = duckdb_v2_schema_get_field(schema, i, &name, &length, &type, err.out());
        if (code != DUCKDB_V2_ERROR_NONE) break;
        // Named as result columns are, by their v1 type.
        duckdb_logical_type logical = duckdb_v2_logical_type_to_v1(type);
        out.emplace_back(std::string(name, length), duckdb_type_name(duckdb_get_type_id(logical)));
        duckdb_destroy_logical_type(&logical);
    }
    duckdb_v2_schema_destroy(&schema);
    err.check(code);
    return out;
}

// Retire a successfully executed statement under the cache policy. Writes do
// not flush the cache: DuckDB re-binds cached plans that read base tables, so
// they see the new data and statistics. Statements that change settings,
// schemas or loaded extensions (everything caches_statement rejects) do.
void Connection::statement_cache_retire(const std::string& query,
                                        std::unique_ptr<V2Prepared> stmt) {
    bool cacheable = caches_statement(stmt->type());
    if (cacheable && handle()) {
        statement_cache_checkin(query, std::move(stmt));
        return;
    }
    // Also taken when close() ran during the execution: never cache statements
    // of a closed connection.
    if (!cacheable) statement_cache_clear();
}

std::unique_ptr<V2Prepared> Connection::statement_cache_checkout(const std::string& query) {
    auto it = statement_cache_.find(query);
    if (it == statement_cache_.end()) return nullptr;
    std::unique_ptr<V2Prepared> stmt = std::move(it->second->stmt);
    statement_lru_.erase(it->second);
    statement_cache_.erase(it);
    return stmt;
}

void Connection::statement_cache_checkin(const std::string& query,
                                         std::unique_ptr<V2Prepared> stmt) {
    // Destroying a statement releases the GIL, letting another thread use the
    // cache: unlink entries before destroying their statements.
    std::unique_ptr<V2Prepared> replaced;
    auto it = statement_cache_.find(query);
    if (it != statement_cache_.end()) {
        // Replace a concurrently returned copy.
        replaced = std::move(it->second->stmt);
        statement_lru_.erase(it->second);
        statement_cache_.erase(it);
    }
    statement_lru_.push_front(CachedStatement{query, std::move(stmt)});
    statement_cache_.emplace(query, statement_lru_.begin());
    std::unique_ptr<V2Prepared> evicted;
    if (statement_lru_.size() > kStatementCacheCapacity) {
        evicted = std::move(statement_lru_.back().stmt);
        statement_cache_.erase(statement_lru_.back().query);
        statement_lru_.pop_back();
    }
}

void Connection::statement_cache_clear() {
    std::list<CachedStatement> entries = std::move(statement_lru_);
    statement_lru_.clear();
    statement_cache_.clear();
}

Connection::StatementLease::StatementLease(Connection& con, duckdb_v2_connection_handle connection,
                                           const std::string& query)
    : con_(con), query_(query) {
    stmt_ = con_.statement_cache_checkout(query);
    if (!stmt_) stmt_ = std::make_unique<V2Prepared>(connection, query);
}

void Connection::StatementLease::retire() { con_.statement_cache_retire(query_, std::move(stmt_)); }

std::shared_ptr<Result> Connection::run(const std::string& query, nb::object parameters,
                                        bool streaming) {
    // Held for the whole execution: a concurrent close() must not disconnect
    // underneath the running query.
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    duckdb_v2_connection_handle con = handle->connection_v2;

    StatementLease stmt(*this, con, query);
    Execution execution(execute_prepared(con, stmt.get(), parameters), streaming);
    drive(con, execution);
    stmt.retire();
    // Skip streaming: its query has not finished, so the profile would reflect
    // the previous query.
    if (!streaming) maybe_emit_profile(con, query);
    if (!this->handle()) throw DuckyError("ducky: connection was closed while the query ran");
    return std::make_shared<Result>(execution, std::move(handle));
}

Connection& Connection::execute(const std::string& query, nb::object parameters, bool streaming) {
    // Release the result this one replaces first: DuckDB refuses a new query
    // while a streaming result holds the connection.
    last_result_.reset();
    last_result_ = run(query, parameters, streaming);
    return *this;
}

std::shared_ptr<Result> Connection::sql(const std::string& query, nb::object parameters,
                                        bool streaming) {
    return run(query, parameters, streaming);
}

void Connection::executemany(const std::string& query, nb::object seq) {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    StatementLease stmt(*this, handle->connection_v2, query);
    executemany_loop(handle->connection_v2, stmt.get(), seq);
    stmt.retire();
}

nb::object Connection::arrow_collected_stream(const std::string& query, nb::object parameters,
                                              uint64_t rows_per_batch) {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    duckdb_v2_connection_handle con = handle->connection_v2;

    StatementLease stmt(*this, con, query);
    V2Parameters params(connection_factory(con), parameters, &stmt.get());
    // An Arrow result's workers assemble the Arrow arrays during execution.
    auto collected = std::make_unique<CollectedArrow>();
    collected->handle = handle;
    V2Error err;
    DUCKDB_V2_ERROR code;
    {
        release_gil_if_held release;
        code = duckdb_v2_prepared_statement_execute_arrow(
            stmt.get().get(), params.names(), params.name_lengths(), params.values(),
            params.count(), rows_per_batch, &collected->result, err.out());
    }
    err.check(code);
    // Read to its end anyway; see Execution.
    duckdb_v2_arrow_result_retain(collected->result, nullptr);
    collected->arrays = collect_arrow(con, collected->result);
    stmt.retire();

    auto* stream = (ArrowArrayStream*)calloc(1, sizeof(ArrowArrayStream));
    if (!stream) throw std::bad_alloc();
    stream->get_schema = &CollectedArrow::get_schema;
    stream->get_next = &CollectedArrow::get_next;
    stream->get_last_error = &CollectedArrow::get_last_error;
    stream->release = &CollectedArrow::release;
    stream->private_data = collected.release();
    PyObject* capsule =
        PyCapsule_New(stream, "arrow_array_stream", arrow_stream_capsule_destructor);
    if (!capsule) {
        stream->release(stream);
        free(stream);
        throw nb::python_error();
    }
    return nb::steal(capsule);
}

PendingResult::PendingResult(std::unique_ptr<Execution> execution,
                             std::shared_ptr<DuckDBHandle> handle)
    : execution_(std::move(execution)), handle_(std::move(handle)) {}

PendingResult::~PendingResult() {
    // Cancel an abandoned coroutine. Ending the query waits for workers, which
    // may need the GIL for Python callbacks.
    if (execution_ && !execution_->settled()) {
        release_gil_if_held release;
        cancel_locked();
    }
}

void PendingResult::cancel_locked() {
    if (!execution_) return;
    duckdb_v2_connection_interrupt(handle_->connection_v2, nullptr);
    execution_.reset();
}

PendingState PendingResult::run_locked(std::chrono::steady_clock::time_point deadline) {
    if (!execution_) throw DuckyError("ducky: pending result already consumed");
    if (!error_.empty()) return PendingState::ERROR;
    try {
        return execution_->run(deadline) ? PendingState::READY : PendingState::NOT_READY;
    } catch (const DuckyError& e) {
        error_ = e.what();
        return PendingState::ERROR;
    }
}

PendingState PendingResult::execute_tasks(double budget) {
    auto span = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(budget > 0 ? budget : 0));
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    return run_locked(std::chrono::steady_clock::now() + span);
}

PendingState PendingResult::execute_task() {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    // A deadline that has passed runs exactly one step.
    return run_locked(std::chrono::steady_clock::time_point::min());
}

std::string PendingResult::error() {
    std::lock_guard<std::mutex> lock(mu_);
    return error_;
}

std::shared_ptr<Result> PendingResult::materialize() {
    std::unique_ptr<Execution> execution;
    {
        nb::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mu_);
        PendingState state;
        do {
            state = run_locked(std::chrono::steady_clock::time_point::max());
        } while (state == PendingState::NOT_READY);
        if (state == PendingState::ERROR) throw DuckyError(error_);
        execution = std::move(execution_);
    }
    return std::make_shared<Result>(*execution, handle_);
}

void PendingResult::drain() {
    // Interrupt before waiting for an in-flight tick.
    duckdb_v2_connection_interrupt(handle_->connection_v2, nullptr);
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    cancel_locked();
}

PendingResult* Connection::make_pending(const std::string& query, nb::object parameters,
                                        bool streaming) {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    duckdb_v2_connection_handle con = handle->connection_v2;

    V2Parameters params(connection_factory(con), parameters);
    ParsedStatement stmt(con, query);
    V2Error err;
    duckdb_v2_result_handle result = nullptr;
    DUCKDB_V2_ERROR code;
    {
        // Binding and planning take the client context lock.
        release_gil_if_held release;
        code = duckdb_v2_statement_execute(con, stmt.get(), params.names(), params.name_lengths(),
                                           params.values(), params.count(), &result, err.out());
    }
    err.check(code);
    auto execution = std::make_unique<Execution>(result, streaming);
    return new PendingResult(std::move(execution), std::move(handle));
}

PreparedStatement* Connection::prepare(const std::string& query) {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    // Read first: a registration racing the prepare makes it prepare again.
    uint64_t generation = handle->arrow_generation.load();
    auto stmt = std::make_unique<V2Prepared>(handle->connection_v2, query);
    return new PreparedStatement(std::move(stmt), std::move(handle), query, generation);
}

std::shared_ptr<Result> Connection::current_result() {
    if (!last_result_) throw DuckyError("ducky: no result set; call execute() first");
    return last_result_;
}

nb::object Connection::fetchone() { return current_result()->fetchone(); }
nb::list Connection::fetchmany(int64_t size) { return current_result()->fetchmany(size); }
nb::list Connection::fetchall() { return current_result()->fetchall(); }
nb::object Connection::fetchitem() { return current_result()->fetchitem(); }

nb::object Connection::description() const {
    if (!last_result_) return nb::none();
    return last_result_->description();
}

nb::object Connection::columns() const {
    if (!last_result_) return nb::none();
    return nb::cast(last_result_->column_names());
}

PreparedStatement::PreparedStatement(std::unique_ptr<V2Prepared> stmt,
                                     std::shared_ptr<DuckDBHandle> handle, std::string query,
                                     uint64_t arrow_generation)
    : stmt_(std::move(stmt)),
      handle_(std::move(handle)),
      query_(std::move(query)),
      arrow_generation_(arrow_generation) {}

PreparedStatement::~PreparedStatement() = default;

std::unique_lock<std::mutex> PreparedStatement::lock_valid() const {
    std::unique_lock<std::mutex> lock = lock_releasing_gil(mu_);
    if (!stmt_) throw DuckyError("ducky: prepared statement is closed");
    uint64_t generation = handle_->arrow_generation.load();
    if (generation != arrow_generation_) {
        stmt_ = std::make_unique<V2Prepared>(handle_->connection_v2, query_);
        arrow_generation_ = generation;
    }
    return lock;
}

std::shared_ptr<Result> PreparedStatement::execute(nb::object parameters, bool streaming) {
    auto lock = lock_valid();
    duckdb_v2_connection_handle con = handle_->connection_v2;
    Execution execution(execute_prepared(con, *stmt_, parameters), streaming);
    drive(con, execution);
    return std::make_shared<Result>(execution, handle_);
}

void PreparedStatement::executemany(nb::object seq) {
    auto lock = lock_valid();
    executemany_loop(handle_->connection_v2, *stmt_, seq);
}

int64_t PreparedStatement::num_parameters() const {
    auto lock = lock_valid();
    return (int64_t)stmt_->parameter_names().size();
}

nb::object PreparedStatement::parameter_name(int64_t index) const {
    auto lock = lock_valid();
    const std::vector<std::string>& names = stmt_->parameter_names();
    if (index < 1 || index > (int64_t)names.size()) return nb::none();
    return nb::str(names[(size_t)index - 1].c_str());
}

std::vector<std::string> PreparedStatement::column_names() const {
    auto lock = lock_valid();
    std::vector<std::string> out;
    for (auto& [name, type] : stmt_->columns(handle_->connection_v2)) out.push_back(name);
    return out;
}

std::vector<std::string> PreparedStatement::column_types() const {
    auto lock = lock_valid();
    std::vector<std::string> out;
    for (auto& [name, type] : stmt_->columns(handle_->connection_v2)) out.push_back(type);
    return out;
}

std::string PreparedStatement::statement_type() const {
    auto lock = lock_valid();
    return statement_type_name(stmt_->type());
}

void PreparedStatement::close() {
    auto lock = lock_releasing_gil(mu_);
    stmt_.reset();
    handle_.reset();
}
