#include "connection.hpp"

#include <datetime.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

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

// Forward declaration so we can recurse for numpy-scalar .item() values.
void bind_one(duckdb_prepared_statement stmt, idx_t idx, nb::handle value);

void bind_check(duckdb_state state, idx_t idx) {
    if (state == DuckDBError) {
        throw DuckyError("ducky: failed to bind parameter " + std::to_string(idx));
    }
}

// Bind a decimal.Decimal as an exact DECIMAL(width, scale). NaN and infinities
// have no DECIMAL form and bind as DOUBLE.
void bind_decimal(duckdb_prepared_statement stmt, idx_t idx, nb::handle value) {
    nb::object parts = value.attr("as_tuple")();
    nb::object exponent = parts.attr("exponent");
    if (!nb::isinstance<nb::int_>(exponent)) {  // 'n', 'N' or 'F'
        bind_check(duckdb_bind_double(stmt, idx, nb::cast<double>(nb::float_(value))), idx);
        return;
    }
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
    duckdb_decimal d{(uint8_t)width, (uint8_t)scale, duckdb_hugeint{lo, (int64_t)hi}};
    bind_check(duckdb_bind_decimal(stmt, idx, d), idx);
}

// Match the planned width to avoid DuckDB replanning on each execute.
void bind_int64_as_param_type(duckdb_prepared_statement stmt, idx_t idx, int64_t v) {
    switch (duckdb_param_type(stmt, idx)) {
        case DUCKDB_TYPE_TINYINT:
            if (v >= INT8_MIN && v <= INT8_MAX) {
                bind_check(duckdb_bind_int8(stmt, idx, (int8_t)v), idx);
                return;
            }
            break;
        case DUCKDB_TYPE_SMALLINT:
            if (v >= INT16_MIN && v <= INT16_MAX) {
                bind_check(duckdb_bind_int16(stmt, idx, (int16_t)v), idx);
                return;
            }
            break;
        case DUCKDB_TYPE_INTEGER:
            if (v >= INT32_MIN && v <= INT32_MAX) {
                bind_check(duckdb_bind_int32(stmt, idx, (int32_t)v), idx);
                return;
            }
            break;
        case DUCKDB_TYPE_UTINYINT:
            if (v >= 0 && v <= UINT8_MAX) {
                bind_check(duckdb_bind_uint8(stmt, idx, (uint8_t)v), idx);
                return;
            }
            break;
        case DUCKDB_TYPE_USMALLINT:
            if (v >= 0 && v <= UINT16_MAX) {
                bind_check(duckdb_bind_uint16(stmt, idx, (uint16_t)v), idx);
                return;
            }
            break;
        case DUCKDB_TYPE_UINTEGER:
            if (v >= 0 && v <= (int64_t)UINT32_MAX) {
                bind_check(duckdb_bind_uint32(stmt, idx, (uint32_t)v), idx);
                return;
            }
            break;
        case DUCKDB_TYPE_UBIGINT:
            if (v >= 0) {
                bind_check(duckdb_bind_uint64(stmt, idx, (uint64_t)v), idx);
                return;
            }
            break;
        case DUCKDB_TYPE_HUGEINT: {
            // Any int64 fits; sign-extend into the upper half.
            duckdb_hugeint h{(uint64_t)v, v < 0 ? (int64_t)-1 : (int64_t)0};
            bind_check(duckdb_bind_hugeint(stmt, idx, h), idx);
            return;
        }
        default:
            // BIGINT, DUCKDB_TYPE_INVALID (type not resolved at plan time),
            // and everything a BIGINT must be cast into (DECIMAL, DOUBLE, ...).
            break;
    }
    bind_check(duckdb_bind_int64(stmt, idx, v), idx);
}

void bind_one(duckdb_prepared_statement stmt, idx_t idx, nb::handle value) {
    const BindTypes& T = bind_types();

    if (value.is_none()) {
        bind_check(duckdb_bind_null(stmt, idx), idx);
        return;
    }
    if (nb::isinstance<nb::bool_>(value)) {
        bind_check(duckdb_bind_boolean(stmt, idx, nb::cast<bool>(value)), idx);
        return;
    }
    if (nb::isinstance<nb::int_>(value)) {
        // PyLong_AsLongLong reports overflow via PyErr_Occurred — no
        // nb::cast_error throw on the common path.
        int64_t v = PyLong_AsLongLong(value.ptr());
        if (v != -1 || !PyErr_Occurred()) {
            bind_int64_as_param_type(stmt, idx, v);
            return;
        }
        PyErr_Clear();
        bool negative = PyObject_RichCompareBool(value.ptr(), nb::int_(0).ptr(), Py_LT) == 1;
        uint64_t lo = 0, hi = 0;
        if (negative) {
            py_long_to_int128(value, /*is_signed=*/true, lo, hi);
            duckdb_hugeint h{lo, (int64_t)(hi)};
            bind_check(duckdb_bind_hugeint(stmt, idx, h), idx);
        } else {
            py_long_to_int128(value, /*is_signed=*/false, lo, hi);
            duckdb_uhugeint h{lo, hi};
            bind_check(duckdb_bind_uhugeint(stmt, idx, h), idx);
        }
        return;
    }
    if (nb::isinstance<nb::float_>(value)) {
        double v = nb::cast<double>(value);
        // Match FLOAT parameters to avoid replanning.
        if (duckdb_param_type(stmt, idx) == DUCKDB_TYPE_FLOAT) {
            bind_check(duckdb_bind_float(stmt, idx, (float)v), idx);
        } else {
            bind_check(duckdb_bind_double(stmt, idx, v), idx);
        }
        return;
    }
    if (nb::isinstance<nb::str>(value)) {
        // Borrow PyUnicode's cached UTF-8 buffer; no std::string copy.
        Py_ssize_t size = 0;
        const char* data = PyUnicode_AsUTF8AndSize(value.ptr(), &size);
        if (!data) throw nb::python_error();
        bind_check(duckdb_bind_varchar_length(stmt, idx, data, (idx_t)size), idx);
        return;
    }
    if (nb::isinstance<nb::bytes>(value)) {
        char* data = nullptr;
        Py_ssize_t size = 0;
        if (PyBytes_AsStringAndSize(value.ptr(), &data, &size) != 0) throw nb::python_error();
        bind_check(duckdb_bind_blob(stmt, idx, data, (idx_t)size), idx);
        return;
    }
    // datetime.datetime is a subclass of datetime.date, so check it first.
    // PyDateTime_* macros read directly from the struct — no attr lookups.
    PyObject* raw = value.ptr();
    if (PyDateTime_Check(raw)) {
        bool aware = !value.attr("tzinfo").is_none();
        if (aware) {
            duckdb_timestamp ts{aware_datetime_to_utc_micros(value)};
            bind_check(duckdb_bind_timestamp_tz(stmt, idx, ts), idx);
        } else {
            duckdb_timestamp_struct s{};
            s.date.year = PyDateTime_GET_YEAR(raw);
            s.date.month = (int8_t)PyDateTime_GET_MONTH(raw);
            s.date.day = (int8_t)PyDateTime_GET_DAY(raw);
            s.time.hour = (int8_t)PyDateTime_DATE_GET_HOUR(raw);
            s.time.min = (int8_t)PyDateTime_DATE_GET_MINUTE(raw);
            s.time.sec = (int8_t)PyDateTime_DATE_GET_SECOND(raw);
            s.time.micros = PyDateTime_DATE_GET_MICROSECOND(raw);
            bind_check(duckdb_bind_timestamp(stmt, idx, duckdb_to_timestamp(s)), idx);
        }
        return;
    }
    if (PyDate_Check(raw)) {
        duckdb_date_struct s{};
        s.year = PyDateTime_GET_YEAR(raw);
        s.month = (int8_t)PyDateTime_GET_MONTH(raw);
        s.day = (int8_t)PyDateTime_GET_DAY(raw);
        bind_check(duckdb_bind_date(stmt, idx, duckdb_to_date(s)), idx);
        return;
    }
    if (PyTime_Check(raw)) {
        duckdb_time_struct s{};
        s.hour = (int8_t)PyDateTime_TIME_GET_HOUR(raw);
        s.min = (int8_t)PyDateTime_TIME_GET_MINUTE(raw);
        s.sec = (int8_t)PyDateTime_TIME_GET_SECOND(raw);
        s.micros = PyDateTime_TIME_GET_MICROSECOND(raw);
        bind_check(duckdb_bind_time(stmt, idx, duckdb_to_time(s)), idx);
        return;
    }
    if (PyDelta_Check(raw)) {
        duckdb_interval iv{};
        iv.months = 0;
        iv.days = PyDateTime_DELTA_GET_DAYS(raw);
        int64_t seconds = PyDateTime_DELTA_GET_SECONDS(raw);
        int32_t micros = PyDateTime_DELTA_GET_MICROSECONDS(raw);
        iv.micros = seconds * 1000000 + micros;
        bind_check(duckdb_bind_interval(stmt, idx, iv), idx);
        return;
    }
    if (nb::isinstance(value, T.decimal)) {
        bind_decimal(stmt, idx, value);
        return;
    }
    if (nb::isinstance(value, T.uuid)) {
        // UUID.int is the natural unsigned 128-bit value duckdb_create_uuid takes.
        uint64_t lo = 0, hi = 0;
        py_long_to_int128(value.attr("int"), /*is_signed=*/false, lo, hi);
        duckdb_value uuid = duckdb_create_uuid(duckdb_uhugeint{lo, hi});
        duckdb_state state = duckdb_bind_value(stmt, idx, uuid);
        duckdb_destroy_value(&uuid);
        bind_check(state, idx);
        return;
    }
    // Unbox NumPy scalars.
    if (nb::hasattr(value, "item")) {
        nb::object item;
        try {
            item = value.attr("item")();
        } catch (...) {
            item = nb::object();
        }
        if (item) {
            if (item.type().ptr() != value.type().ptr()) {
                bind_one(stmt, idx, item);
                return;
            }
        }
    }

    throw DuckyError(std::string("ducky: unsupported parameter type at position ") +
                     std::to_string(idx) + " (" + nb::cast<std::string>(nb::str(value.type())) +
                     ")");
}

void bind_positional(duckdb_prepared_statement stmt, nb::handle parameters) {
    idx_t param = 0;
    for (nb::handle value : parameters) {
        param += 1;  // DuckDB params are 1-indexed.
        bind_one(stmt, param, value);
    }
}

void bind_named(duckdb_prepared_statement stmt, nb::handle parameters) {
    nb::dict d = nb::borrow<nb::dict>(parameters);
    for (auto [k, v] : d) {
        std::string name = nb::cast<std::string>(k);
        idx_t idx = 0;
        if (duckdb_bind_parameter_index(stmt, &idx, name.c_str()) == DuckDBError) {
            throw DuckyError("ducky: unknown named parameter '" + name + "'");
        }
        bind_one(stmt, idx, v);
    }
}

void bind_parameters(duckdb_prepared_statement stmt, nb::handle parameters) {
    if (nb::isinstance<nb::dict>(parameters)) {
        bind_named(stmt, parameters);
    } else {
        bind_positional(stmt, parameters);
    }
}

// Shared executemany loop with periodic signal checks.
void executemany_loop(duckdb_prepared_statement stmt, nb::handle seq) {
    constexpr idx_t SIGNAL_CHECK_PERIOD = 1024;
    idx_t since_check = 0;
    for (nb::handle params : seq) {
        // Every parameter set must stand alone. Without clearing, a short
        // positional row or a named dict missing a key silently inherits the
        // previous row's binding and writes incorrect data.
        if (duckdb_clear_bindings(stmt) == DuckDBError) {
            throw DuckyError("ducky: failed to clear prepared-statement bindings");
        }
        bind_parameters(stmt, params);
        duckdb_result result;
        duckdb_state state;
        {
            // Execution may run Python UDFs on DuckDB workers.
            nb::gil_scoped_release release;
            state = duckdb_execute_prepared(stmt, &result);
        }
        if (state == DuckDBError) {
            std::string message = duckdb_message(duckdb_result_error(&result));
            duckdb_destroy_result(&result);
            throw DuckyError(message);
        }
        duckdb_destroy_result(&result);
        if (++since_check >= SIGNAL_CHECK_PERIOD) {
            since_check = 0;
            if (PyErr_CheckSignals() != 0) throw nb::python_error();
        }
    }
}

// Drive and consume `pending`, checking Python signals between ticks.
duckdb_result drive_pending(duckdb_connection con, duckdb_pending_result pending,
                            duckdb_state init) {
    if (init == DuckDBError) {
        std::string message = duckdb_message(duckdb_pending_error(pending));
        duckdb_destroy_pending(&pending);
        throw DuckyError(message);
    }

    // While DuckDB's workers finish the last tasks of a pipeline there is
    // nothing to execute here: yield, then sleep briefly. Short sleeps keep the
    // wake-up latency well below a typical query's runtime.
    constexpr int yield_ticks = 64;
    constexpr std::chrono::microseconds min_sleep(20), max_sleep(100);
    int no_tasks_streak = 0;
    std::chrono::microseconds backoff(0);
    for (;;) {
        duckdb_pending_state state;
        {
            // Release the GIL and resurface every 10 ms for signals.
            nb::gil_scoped_release release;
            auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(10);
            for (;;) {
                state = duckdb_pending_execute_task(pending);
                if (state == DUCKDB_PENDING_RESULT_READY || state == DUCKDB_PENDING_ERROR) break;
                if (state == DUCKDB_PENDING_NO_TASKS_AVAILABLE) {
                    if (++no_tasks_streak > yield_ticks) {
                        backoff = std::min(std::max(backoff * 2, min_sleep), max_sleep);
                        std::this_thread::sleep_for(backoff);
                    } else {
                        std::this_thread::yield();
                    }
                } else {
                    no_tasks_streak = 0;
                    backoff = std::chrono::microseconds(0);
                }
                if (std::chrono::steady_clock::now() >= deadline) break;
            }
        }
        if (state == DUCKDB_PENDING_ERROR) {
            std::string message = duckdb_message(duckdb_pending_error(pending));
            duckdb_destroy_pending(&pending);
            throw DuckyError(message);
        }
        if (PyErr_CheckSignals() != 0) {
            // Typically KeyboardInterrupt. Interrupt, drain, discard, re-raise.
            {
                // The drain waits for workers, which may need the GIL for
                // Python callbacks; the signal's exception stays set meanwhile.
                nb::gil_scoped_release release;
                duckdb_interrupt(con);
                duckdb_result drain;
                duckdb_execute_pending(pending, &drain);
                duckdb_destroy_result(&drain);
                duckdb_destroy_pending(&pending);
            }
            throw nb::python_error();
        }
        if (state == DUCKDB_PENDING_RESULT_READY) break;
        // DUCKDB_PENDING_RESULT_NOT_READY / NO_TASKS_AVAILABLE: keep ticking.
    }

    duckdb_result result;
    duckdb_state final_state = duckdb_execute_pending(pending, &result);
    duckdb_destroy_pending(&pending);
    if (final_state == DuckDBError) {
        std::string message = duckdb_message(duckdb_result_error(&result));
        duckdb_destroy_result(&result);
        throw DuckyError(message);
    }
    return result;
}

// `streaming` selects duckdb_pending_prepared_streaming for the lazy
// chunk-pull result. The caller still owns `stmt`.
duckdb_result run_pending(duckdb_connection con, duckdb_prepared_statement stmt, bool streaming) {
    duckdb_pending_result pending = nullptr;
    duckdb_state init = streaming ? duckdb_pending_prepared_streaming(stmt, &pending)
                                  : duckdb_pending_prepared(stmt, &pending);
    return drive_pending(con, pending, init);
}

void arrow_stream_capsule_destructor(PyObject* capsule) {
    auto* stream = (ArrowArrayStream*)PyCapsule_GetPointer(capsule, "arrow_array_stream");
    if (stream) {
        if (stream->release) stream->release(stream);
        free(stream);
    }
}

// Move an Arrow-collected result into a stream capsule.
nb::object arrow_stream_capsule_from_result(duckdb_result* result) {
    auto* stream = (ArrowArrayStream*)calloc(1, sizeof(ArrowArrayStream));
    if (!stream) {
        duckdb_destroy_result(result);
        throw std::bad_alloc();
    }
    std::string error;
    bool failed = drain_arrow_error(duckdb_result_arrow_array_stream(result, stream), error);
    duckdb_destroy_result(result);
    if (failed) {
        free(stream);
        throw DuckyError("ducky: " + error);
    }
    PyObject* capsule =
        PyCapsule_New(stream, "arrow_array_stream", arrow_stream_capsule_destructor);
    if (!capsule) {
        stream->release(stream);
        free(stream);
        throw nb::python_error();
    }
    return nb::steal(capsule);
}

}  // namespace

Connection::Connection(const std::string& database, nb::object config) {
    auto handle = std::make_shared<DuckDBHandle>();

    // Iterate via .items() so we accept any read-only Mapping (TypedDict,
    // MappingProxy, user mappings), not just nb::dict.
    duckdb_config cfg = nullptr;
    if (!config.is_none()) {
        if (duckdb_create_config(&cfg) == DuckDBError) {
            throw DuckyError("ducky: failed to allocate duckdb_config");
        }
        try {
            for (nb::handle item : config.attr("items")()) {
                nb::tuple kv = nb::cast<nb::tuple>(item);
                std::string name = nb::cast<std::string>(kv[0]);
                std::string value = nb::cast<std::string>(kv[1]);
                if (duckdb_set_config(cfg, name.c_str(), value.c_str()) == DuckDBError) {
                    throw DuckyError("ducky: invalid config option '" + name + "' = '" + value +
                                     "'");
                }
            }
        } catch (...) {
            duckdb_destroy_config(&cfg);
            throw;
        }
    }

    char* error = nullptr;
    duckdb_state state = duckdb_open_ext(database.c_str(), &handle->database, cfg, &error);
    duckdb_destroy_config(&cfg);
    if (state == DuckDBError) {
        std::string message = error ? error : "unknown error";
        duckdb_free(error);
        // ~DuckDBHandle closes the (opened-or-not) database as `handle` unwinds.
        throw DuckyError("ducky: failed to open '" + database + "': " + message);
    }
    if (duckdb_connect(handle->database, &handle->connection) == DuckDBError) {
        throw DuckyError("ducky: failed to connect to '" + database + "'");
    }
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
        if (handle->connection) duckdb_interrupt(handle->connection);
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
    std::string sql = "LOAD '" + literal + "'";

    duckdb_result result;
    duckdb_state state;
    {
        // dlopen + metadata/signature checks touch the filesystem; release the
        // GIL so other threads run during the load.
        nb::gil_scoped_release release;
        state = duckdb_query(handle->connection, sql.c_str(), &result);
    }
    if (state == DuckDBError) {
        const char* err = duckdb_result_error(&result);
        std::string message = err ? err : "unknown error";
        duckdb_destroy_result(&result);
        throw DuckyError("ducky: failed to load extension '" + name + "': " + message);
    }
    duckdb_destroy_result(&result);

    // LOAD can register new functions / types / settings, so drop cached plans
    // — mirrors the caches_statement() flush run() applies to LOAD statements.
    statement_cache_clear();
}

nb::tuple Connection::progress() const {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    duckdb_query_progress_type p = duckdb_query_progress(handle->connection);
    return nb::make_tuple(p.percentage, p.rows_processed, p.total_rows_to_process);
}

namespace {

// Recursively materialize a duckdb_profiling_info node into a Python dict.
// The info nodes themselves are borrowed (owned by the connection's profiler
// state); only the duckdb_value handles for metric keys/values need destroying.
nb::dict walk_profiling_node(duckdb_profiling_info info) {
    nb::dict node;
    nb::dict metrics;
    duckdb_value map = duckdb_profiling_info_get_metrics(info);
    if (map) {
        idx_t n = duckdb_get_map_size(map);
        for (idx_t i = 0; i < n; ++i) {
            duckdb_value k = duckdb_get_map_key(map, i);
            duckdb_value v = duckdb_get_map_value(map, i);
            char* ks = duckdb_get_varchar(k);
            char* vs = duckdb_get_varchar(v);
            metrics[nb::str(ks ? ks : "")] = nb::str(vs ? vs : "");
            duckdb_free(ks);
            duckdb_free(vs);
            duckdb_destroy_value(&k);
            duckdb_destroy_value(&v);
        }
        duckdb_destroy_value(&map);
    }
    node["metrics"] = std::move(metrics);

    nb::list children;
    idx_t cc = duckdb_profiling_info_get_child_count(info);
    for (idx_t i = 0; i < cc; ++i) {
        children.append(walk_profiling_node(duckdb_profiling_info_get_child(info, i)));
    }
    node["children"] = std::move(children);
    return node;
}

}  // namespace

nb::object Connection::get_profiling_info() const {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    duckdb_profiling_info info = duckdb_get_profiling_info(handle->connection);
    if (!info) return nb::none();
    return walk_profiling_node(info);
}

namespace {

// Run a SET (or other no-result) statement, bypassing run() so it doesn't
// reach the profile sink.
void exec_no_result(duckdb_connection con, const char* sql) {
    duckdb_result r;
    duckdb_state state;
    {
        // Takes the client context lock, which a query running Python
        // callbacks on another thread may hold while it waits for the GIL.
        nb::gil_scoped_release release;
        state = duckdb_query(con, sql, &r);
    }
    if (state == DuckDBError) {
        std::string message = duckdb_message(duckdb_result_error(&r));
        duckdb_destroy_result(&r);
        throw DuckyError("ducky: " + std::string(sql) + " failed: " + message);
    }
    duckdb_destroy_result(&r);
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
    duckdb_connection con = handle->connection;
    exec_no_result(con, "SET enable_profiling='no_output'");
    // profiling_mode persists, so always set it.
    exec_no_result(con, ("SET profiling_mode='" + mode + "'").c_str());
    profile_sink_ = sink;
    profile_sample_ = sample;
    profile_counter_ = 0;
}

void Connection::maybe_emit_profile(duckdb_connection connection, const std::string& query) {
    // Fast path: no sink installed.
    if (profile_sink_.is_none()) return;
    int64_t n = profile_counter_++;
    if (n % profile_sample_ != 0) return;
    duckdb_profiling_info info = duckdb_get_profiling_info(connection);
    if (!info) return;
    nb::object tree = walk_profiling_node(info);
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
    if (handle && handle->connection) duckdb_interrupt(handle->connection);
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
    if (arrow_registry_) {
        std::lock_guard<std::mutex> lock(arrow_registry_->mu);
        arrow_registry_->sources.clear();
    }
    profile_sink_ = nb::none();
}

namespace {

constexpr size_t kStatementCacheCapacity = 128;

// Cache queries whose plans remain valid across repeated use.
bool caches_statement(duckdb_statement_type type) {
    switch (type) {
        case DUCKDB_STATEMENT_TYPE_SELECT:
        case DUCKDB_STATEMENT_TYPE_INSERT:
        case DUCKDB_STATEMENT_TYPE_UPDATE:
        case DUCKDB_STATEMENT_TYPE_DELETE:
        case DUCKDB_STATEMENT_TYPE_EXPLAIN:
        case DUCKDB_STATEMENT_TYPE_COPY:
        case DUCKDB_STATEMENT_TYPE_ANALYZE:
        case DUCKDB_STATEMENT_TYPE_TRANSACTION:
        case DUCKDB_STATEMENT_TYPE_RELATION:
            return true;
        default:
            return false;
    }
}

// Destroy a prepared statement. Takes the client context lock, which a query
// running Python callbacks on another thread may hold while it waits for the
// GIL.
void destroy_prepared(duckdb_prepared_statement& stmt) {
    if (!stmt) return;
    release_gil_if_held release;
    duckdb_destroy_prepare(&stmt);
}

}  // namespace

// Retire a successfully executed statement under the cache policy. Writes do
// not flush the cache: DuckDB re-binds cached plans that read base tables, so
// they see the new data and statistics. Statements that change settings,
// schemas or loaded extensions (everything caches_statement rejects) do.
void Connection::statement_cache_retire(const std::string& query, duckdb_prepared_statement stmt) {
    duckdb_statement_type type = duckdb_prepared_statement_type(stmt);
    if (caches_statement(type) && handle()) {
        statement_cache_checkin(query, stmt);
    } else {
        // Also taken when close() ran during the execution: never cache
        // statements of a closed connection.
        if (!caches_statement(type)) statement_cache_clear();
        destroy_prepared(stmt);
    }
}

duckdb_prepared_statement Connection::statement_cache_checkout(const std::string& query) {
    auto it = statement_cache_.find(query);
    if (it == statement_cache_.end()) return nullptr;
    duckdb_prepared_statement stmt = it->second->stmt;
    statement_lru_.erase(it->second);
    statement_cache_.erase(it);
    return stmt;
}

void Connection::statement_cache_checkin(const std::string& query, duckdb_prepared_statement stmt) {
    // destroy_prepared releases the GIL, letting another thread use the cache:
    // unlink entries before destroying their statements.
    duckdb_prepared_statement replaced = nullptr;
    auto it = statement_cache_.find(query);
    if (it != statement_cache_.end()) {
        // Replace a concurrently returned copy.
        replaced = it->second->stmt;
        statement_lru_.erase(it->second);
        statement_cache_.erase(it);
    }
    statement_lru_.push_front(CachedStatement{query, stmt});
    statement_cache_.emplace(query, statement_lru_.begin());
    duckdb_prepared_statement evicted = nullptr;
    if (statement_lru_.size() > kStatementCacheCapacity) {
        evicted = statement_lru_.back().stmt;
        statement_cache_.erase(statement_lru_.back().query);
        statement_lru_.pop_back();
    }
    destroy_prepared(replaced);
    destroy_prepared(evicted);
}

void Connection::statement_cache_clear() {
    std::list<CachedStatement> entries = std::move(statement_lru_);
    statement_lru_.clear();
    statement_cache_.clear();
    for (CachedStatement& entry : entries) destroy_prepared(entry.stmt);
}

namespace {

// Prepare `query`, raising DuckDB's message on failure. Runs without the GIL:
// preparing takes the client context lock.
duckdb_prepared_statement prepare_statement(duckdb_connection connection,
                                            const std::string& query) {
    duckdb_prepared_statement stmt = nullptr;
    duckdb_state state;
    {
        release_gil_if_held release;
        state = duckdb_prepare(connection, query.c_str(), &stmt);
    }
    if (state == DuckDBError) {
        std::string message = duckdb_message(duckdb_prepare_error(stmt));
        destroy_prepared(stmt);
        throw DuckyError(message);
    }
    return stmt;
}

}  // namespace

Connection::StatementLease::StatementLease(Connection& con, duckdb_connection connection,
                                           const std::string& query)
    : con_(con), query_(query) {
    stmt_ = con_.statement_cache_checkout(query);
    // Cached statements must start unbound.
    if (stmt_ && duckdb_clear_bindings(stmt_) == DuckDBError) destroy_prepared(stmt_);
    if (!stmt_) stmt_ = prepare_statement(connection, query);
}

Connection::StatementLease::~StatementLease() {
    // Do not cache statements whose execution threw.
    destroy_prepared(stmt_);
}

void Connection::StatementLease::retire() {
    con_.statement_cache_retire(query_, stmt_);
    stmt_ = nullptr;
}

duckdb_result Connection::run(const std::string& query, nb::object parameters, bool streaming) {
    // Held for the whole execution: a concurrent close() must not disconnect
    // underneath the running query.
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    duckdb_connection con = handle->connection;

    StatementLease stmt(*this, con, query);
    if (!parameters.is_none()) bind_parameters(stmt.get(), parameters);
    duckdb_result result = run_pending(con, stmt.get(), streaming);
    stmt.retire();
    // Skip streaming: chunks haven't been pulled when run_pending returns,
    // so the profile would reflect the previous query.
    if (!streaming) {
        try {
            maybe_emit_profile(con, query);
        } catch (...) {
            duckdb_destroy_result(&result);
            throw;
        }
    }
    return result;
}

std::shared_ptr<Result> Connection::make_result(duckdb_result result) {
    std::shared_ptr<DuckDBHandle> handle = this->handle();
    if (!handle) {
        // close() ran while the query executed.
        duckdb_destroy_result(&result);
        throw DuckyError("ducky: connection was closed while the query ran");
    }
    return std::make_shared<Result>(result, std::move(handle));
}

Connection& Connection::execute(const std::string& query, nb::object parameters, bool streaming) {
    // Release the result this one replaces first: DuckDB refuses a new query
    // while a streaming result holds the connection.
    last_result_.reset();
    last_result_ = make_result(run(query, parameters, streaming));
    return *this;
}

std::shared_ptr<Result> Connection::sql(const std::string& query, nb::object parameters,
                                        bool streaming) {
    return make_result(run(query, parameters, streaming));
}

void Connection::executemany(const std::string& query, nb::object seq) {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    StatementLease stmt(*this, handle->connection, query);
    executemany_loop(stmt.get(), seq);
    stmt.retire();
}

nb::object Connection::arrow_collected_stream(const std::string& query, nb::object parameters,
                                              uint64_t rows_per_batch) {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    duckdb_connection con = handle->connection;

    StatementLease stmt(*this, con, query);
    if (!parameters.is_none()) bind_parameters(stmt.get(), parameters);

    // duckdb_pending_prepared_arrow plans the query with DuckDB's Arrow result
    // collector, so execution assembles Arrow batches directly; the normal
    // pending tick loop keeps it Ctrl-C interruptible.
    duckdb_pending_result pending = nullptr;
    duckdb_state init = duckdb_pending_prepared_arrow(stmt.get(), rows_per_batch, &pending);
    duckdb_result result = drive_pending(con, pending, init);
    nb::object capsule = arrow_stream_capsule_from_result(&result);
    stmt.retire();
    return capsule;
}

PendingResult::PendingResult(duckdb_pending_result pending, duckdb_prepared_statement stmt,
                             bool owns_stmt, duckdb_connection con,
                             std::shared_ptr<DuckDBHandle> handle)
    : pending_(pending),
      stmt_(stmt),
      owns_stmt_(owns_stmt),
      con_(con),
      handle_(std::move(handle)) {}

void PendingResult::cleanup_locked() {
    if (pending_) duckdb_destroy_pending(&pending_);
    if (owns_stmt_ && stmt_) duckdb_destroy_prepare(&stmt_);
    pending_ = nullptr;
    stmt_ = nullptr;
}

PendingResult::~PendingResult() {
    // Cancel an abandoned coroutine. The drain waits for workers, which may
    // need the GIL for Python callbacks.
    if (pending_) {
        release_gil_if_held release;
        if (con_) duckdb_interrupt(con_);
        duckdb_result drained;
        duckdb_execute_pending(pending_, &drained);
        duckdb_destroy_result(&drained);
        cleanup_locked();
    }
}

duckdb_pending_state PendingResult::execute_task() {
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    if (!pending_) throw DuckyError("ducky: pending result already consumed");
    return duckdb_pending_execute_task(pending_);
}

std::string PendingResult::error() {
    std::lock_guard<std::mutex> lock(mu_);
    return pending_ ? duckdb_message(duckdb_pending_error(pending_)) : std::string();
}

std::shared_ptr<Result> PendingResult::materialize() {
    duckdb_result result;
    duckdb_state final_state;
    {
        nb::gil_scoped_release release;
        std::lock_guard<std::mutex> lock(mu_);
        if (!pending_) throw DuckyError("ducky: pending result already consumed");
        final_state = duckdb_execute_pending(pending_, &result);
        cleanup_locked();
    }
    if (final_state == DuckDBError) {
        std::string message = duckdb_message(duckdb_result_error(&result));
        duckdb_destroy_result(&result);
        throw DuckyError(message);
    }
    return std::make_shared<Result>(result, handle_);
}

void PendingResult::drain() {
    // Interrupt before waiting for an in-flight tick.
    if (con_) duckdb_interrupt(con_);
    nb::gil_scoped_release release;
    std::lock_guard<std::mutex> lock(mu_);
    if (!pending_) return;
    duckdb_result drained;
    duckdb_execute_pending(pending_, &drained);
    duckdb_destroy_result(&drained);
    cleanup_locked();
}

PendingResult* Connection::make_pending(const std::string& query, nb::object parameters,
                                        bool streaming) {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    duckdb_connection con = handle->connection;

    duckdb_prepared_statement stmt = prepare_statement(con, query);
    if (!parameters.is_none()) {
        try {
            bind_parameters(stmt, parameters);
        } catch (...) {
            destroy_prepared(stmt);
            throw;
        }
    }
    duckdb_pending_result pending = nullptr;
    duckdb_state init;
    {
        release_gil_if_held release;
        init = streaming ? duckdb_pending_prepared_streaming(stmt, &pending)
                         : duckdb_pending_prepared(stmt, &pending);
    }
    if (init == DuckDBError) {
        std::string message = duckdb_message(duckdb_pending_error(pending));
        duckdb_destroy_pending(&pending);
        destroy_prepared(stmt);
        throw DuckyError(message);
    }
    return new PendingResult(pending, stmt, /*owns_stmt=*/true, con, std::move(handle));
}

PreparedStatement* Connection::prepare(const std::string& query) {
    std::shared_ptr<DuckDBHandle> handle = open_handle();
    duckdb_prepared_statement stmt = prepare_statement(handle->connection, query);
    return new PreparedStatement(stmt, std::move(handle));
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

PreparedStatement::PreparedStatement(duckdb_prepared_statement stmt,
                                     std::shared_ptr<DuckDBHandle> handle)
    : stmt_(stmt), handle_(std::move(handle)) {}

PreparedStatement::~PreparedStatement() { destroy_prepared(stmt_); }

std::unique_lock<std::mutex> PreparedStatement::lock_valid() const {
    std::unique_lock<std::mutex> lock = lock_releasing_gil(mu_);
    if (!stmt_) throw DuckyError("ducky: prepared statement is closed");
    return lock;
}

std::shared_ptr<Result> PreparedStatement::execute(nb::object parameters, bool streaming) {
    auto lock = lock_valid();
    // Each execution starts unbound.
    if (duckdb_clear_bindings(stmt_) == DuckDBError) {
        throw DuckyError("ducky: failed to clear prepared-statement bindings");
    }
    if (!parameters.is_none()) {
        bind_parameters(stmt_, parameters);
    }
    duckdb_result result = run_pending(handle_->connection, stmt_, streaming);
    return std::make_shared<Result>(result, handle_);
}

void PreparedStatement::executemany(nb::object seq) {
    auto lock = lock_valid();
    executemany_loop(stmt_, seq);
}

int64_t PreparedStatement::num_parameters() const {
    auto lock = lock_valid();
    return (int64_t)duckdb_nparams(stmt_);
}

nb::object PreparedStatement::parameter_name(int64_t index) const {
    auto lock = lock_valid();
    const char* name = duckdb_parameter_name(stmt_, (idx_t)index);
    if (!name) return nb::none();
    nb::object out = nb::str(name);
    duckdb_free((void*)name);
    return out;
}

std::vector<std::string> PreparedStatement::column_names() const {
    auto lock = lock_valid();
    idx_t n = duckdb_prepared_statement_column_count(stmt_);
    std::vector<std::string> out;
    out.reserve(n);
    for (idx_t i = 0; i < n; ++i) {
        const char* name = duckdb_prepared_statement_column_name(stmt_, i);
        out.emplace_back(name ? name : "");
        if (name) duckdb_free((void*)name);
    }
    return out;
}

std::vector<std::string> PreparedStatement::column_types() const {
    auto lock = lock_valid();
    idx_t n = duckdb_prepared_statement_column_count(stmt_);
    std::vector<std::string> out;
    out.reserve(n);
    for (idx_t i = 0; i < n; ++i) {
        out.emplace_back(duckdb_type_name(duckdb_prepared_statement_column_type(stmt_, i)));
    }
    return out;
}

std::string PreparedStatement::statement_type() const {
    auto lock = lock_valid();
    return duckdb_statement_type_name(duckdb_prepared_statement_type(stmt_));
}

void PreparedStatement::close() {
    auto lock = lock_releasing_gil(mu_);
    destroy_prepared(stmt_);
    handle_.reset();
}
