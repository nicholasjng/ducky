#include "connection.hpp"

#include <datetime.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include "arrow_scan.hpp"
#include "ducky.hpp"
#include "prepared.hpp"

namespace {

using namespace nb::literals;

// Cached Python types and 128-bit constants used by bind_one. Imported once,
// intentionally leaked so the destructors don't run at interpreter shutdown.
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
            bind_check(duckdb_bind_int64(stmt, idx, v), idx);
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
        bind_check(duckdb_bind_double(stmt, idx, nb::cast<double>(value)), idx);
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
    if (nb::isinstance(value, T.decimal) || nb::isinstance(value, T.uuid)) {
        // Bind as VARCHAR; DuckDB implicitly casts to DECIMAL / UUID on execute.
        nb::str s = nb::str(value);
        bind_check(duckdb_bind_varchar(stmt, idx, s.c_str()), idx);
        return;
    }
    // numpy scalars (anything exposing .item() → native Python): re-dispatch
    // on the unboxed value.
    if (nb::hasattr(value, "item")) {
        nb::object item;
        try {
            item = value.attr("item")();
        } catch (...) {
            item = nb::object();
        }
        if (item) {
            // Native Python scalars never expose .item(), so the type-change
            // check terminates immediately — no infinite recursion.
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

// Drive duckdb_pending_execute_task in a GIL-releasing loop until READY (or
// error), then materialize via duckdb_execute_pending. Ticking one task at a
// time (rather than blocking inside duckdb_execute_prepared) lets us run
// PyErr_CheckSignals between ticks (Ctrl-C interrupts mid-query) and reach the
// streaming result path. `streaming` selects duckdb_pending_prepared_streaming
// for the lazy chunk-pull result. The caller still owns `stmt`.
duckdb_result run_pending(duckdb_connection con, duckdb_prepared_statement stmt, bool streaming) {
    duckdb_pending_result pending = nullptr;
    duckdb_state init = streaming ? duckdb_pending_prepared_streaming(stmt, &pending)
                                  : duckdb_pending_prepared(stmt, &pending);
    if (init == DuckDBError) {
        std::string message = duckdb_pending_error(pending);
        duckdb_destroy_pending(&pending);
        throw DuckyError(message);
    }

    for (;;) {
        duckdb_pending_state state;
        {
            nb::gil_scoped_release release;
            state = duckdb_pending_execute_task(pending);
            // Workers own the remaining tasks; yield briefly instead of spinning.
            if (state == DUCKDB_PENDING_NO_TASKS_AVAILABLE) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        if (state == DUCKDB_PENDING_ERROR) {
            std::string message = duckdb_pending_error(pending);
            duckdb_destroy_pending(&pending);
            throw DuckyError(message);
        }
        if (PyErr_CheckSignals() != 0) {
            // Typically KeyboardInterrupt. Interrupt, drain, discard, re-raise.
            duckdb_interrupt(con);
            duckdb_result drain;
            duckdb_execute_pending(pending, &drain);
            duckdb_destroy_result(&drain);
            duckdb_destroy_pending(&pending);
            throw nb::python_error();
        }
        if (state == DUCKDB_PENDING_RESULT_READY) break;
        // DUCKDB_PENDING_RESULT_NOT_READY / NO_TASKS_AVAILABLE: keep ticking.
    }

    duckdb_result result;
    duckdb_state final_state = duckdb_execute_pending(pending, &result);
    duckdb_destroy_pending(&pending);
    if (final_state == DuckDBError) {
        std::string message = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw DuckyError(message);
    }
    return result;
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

void Connection::interrupt() {
    if (handle_ && handle_->connection) duckdb_interrupt(handle_->connection);
}

nb::tuple Connection::progress() const {
    if (!handle_ || !handle_->connection) {
        throw DuckyError("ducky: connection is closed");
    }
    duckdb_query_progress_type p = duckdb_query_progress(handle_->connection);
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
    if (!handle_ || !handle_->connection) {
        throw DuckyError("ducky: connection is closed");
    }
    duckdb_profiling_info info = duckdb_get_profiling_info(handle_->connection);
    if (!info) return nb::none();
    return walk_profiling_node(info);
}

namespace {

// Run a SET (or other no-result) statement, bypassing run() so it doesn't
// reach the profile sink.
void exec_no_result(duckdb_connection con, const char* sql) {
    duckdb_result r;
    if (duckdb_query(con, sql, &r) == DuckDBError) {
        std::string message = duckdb_result_error(&r);
        duckdb_destroy_result(&r);
        throw DuckyError("ducky: " + std::string(sql) + " failed: " + message);
    }
    duckdb_destroy_result(&r);
}

}  // namespace

void Connection::set_profile_sink(nb::object sink, int64_t sample, const std::string& mode) {
    ensure_open();
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
    duckdb_connection con = handle_->connection;
    exec_no_result(con, "SET enable_profiling='no_output'");
    if (mode == "detailed") {
        exec_no_result(con, "SET profiling_mode='detailed'");
    }
    profile_sink_ = sink;
    profile_sample_ = sample;
    profile_counter_ = 0;
}

void Connection::maybe_emit_profile(const std::string& query) {
    // Fast path: no sink installed.
    if (profile_sink_.is_none()) return;
    int64_t n = profile_counter_++;
    if (n % profile_sample_ != 0) return;
    duckdb_profiling_info info = duckdb_get_profiling_info(handle_->connection);
    if (!info) return;
    nb::object tree = walk_profiling_node(info);
    // Sink errors must not break the query — print to stderr and continue.
    try {
        profile_sink_(query, tree);
    } catch (nb::python_error& e) {
        PyErr_WriteUnraisable(profile_sink_.ptr());
    }
}

void Connection::register_arrow(const std::string& name, nb::object obj) {
    ensure_open();
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

    // Lazy, zero-copy: stash the source and resolve `SELECT * FROM name` to the
    // ducky_arrow_scan table function via the replacement scan, re-streaming the
    // source on each query (so it must support being streamed more than once).
    if (!arrow_registry_) arrow_registry_ = install_arrow_scan(*this);
    std::lock_guard<std::mutex> lock(arrow_registry_->mu);
    arrow_registry_->sources[name] = std::move(obj);  // insert / overwrite (GIL held)
}

Connection::~Connection() { close(); }

void Connection::close() {
    // Release this Connection's references. The database stays open until the
    // last Result sharing the handle is also gone.
    last_result_.reset();
    handle_.reset();
}

void Connection::ensure_open() const {
    if (!handle_ || !handle_->connection) throw DuckyError("ducky: connection is closed");
}

duckdb_result Connection::run(const std::string& query, nb::object parameters, bool streaming) {
    ensure_open();
    duckdb_connection con = handle_->connection;

    // Always go through prepare + pending-execute so the run_pending semantics
    // (signal handling, streaming) apply. Single statement only.
    duckdb_prepared_statement stmt = nullptr;
    if (duckdb_prepare(con, query.c_str(), &stmt) == DuckDBError) {
        std::string message = duckdb_prepare_error(stmt);
        duckdb_destroy_prepare(&stmt);
        throw DuckyError(message);
    }
    if (!parameters.is_none()) {
        try {
            bind_parameters(stmt, parameters);
        } catch (...) {
            duckdb_destroy_prepare(&stmt);
            throw;
        }
    }
    duckdb_result result;
    try {
        result = run_pending(con, stmt, streaming);
    } catch (...) {
        duckdb_destroy_prepare(&stmt);
        throw;
    }
    duckdb_destroy_prepare(&stmt);
    // Skip streaming: chunks haven't been pulled when run_pending returns,
    // so the profile would reflect the previous query.
    if (!streaming) {
        maybe_emit_profile(query);
    }
    return result;
}

std::shared_ptr<Result> Connection::make_result(duckdb_result result) {
    return std::make_shared<Result>(result, handle_);
}

Connection& Connection::execute(const std::string& query, nb::object parameters, bool streaming) {
    last_result_ = make_result(run(query, parameters, streaming));
    return *this;
}

std::shared_ptr<Result> Connection::sql(const std::string& query, bool streaming) {
    return make_result(run(query, nb::none(), streaming));
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
    // Backstop for an abandoned coroutine. A live worker would hold a ref to
    // this object via execute_task, so no tick can be in flight — no lock
    // needed. Interrupt first so a partial query stops promptly.
    if (pending_) {
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
    return pending_ ? duckdb_pending_error(pending_) : std::string();
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
        std::string message = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw DuckyError(message);
    }
    return std::make_shared<Result>(result, handle_);
}

void PendingResult::drain() {
    // Interrupt before taking the lock so a mid-tick worker on another thread
    // returns promptly and lets the lock free up. GIL is dropped so the worker
    // can reacquire it to finish.
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
    ensure_open();
    duckdb_connection con = handle_->connection;

    duckdb_prepared_statement stmt = nullptr;
    if (duckdb_prepare(con, query.c_str(), &stmt) == DuckDBError) {
        std::string message = duckdb_prepare_error(stmt);
        duckdb_destroy_prepare(&stmt);
        throw DuckyError(message);
    }
    if (!parameters.is_none()) {
        try {
            bind_parameters(stmt, parameters);
        } catch (...) {
            duckdb_destroy_prepare(&stmt);
            throw;
        }
    }
    duckdb_pending_result pending = nullptr;
    duckdb_state init = streaming ? duckdb_pending_prepared_streaming(stmt, &pending)
                                  : duckdb_pending_prepared(stmt, &pending);
    if (init == DuckDBError) {
        std::string message = duckdb_pending_error(pending);
        duckdb_destroy_pending(&pending);
        duckdb_destroy_prepare(&stmt);
        throw DuckyError(message);
    }
    return new PendingResult(pending, stmt, /*owns_stmt=*/true, con, handle_);
}

PreparedStatement* Connection::prepare(const std::string& query) {
    ensure_open();
    duckdb_prepared_statement stmt = nullptr;
    if (duckdb_prepare(handle_->connection, query.c_str(), &stmt) == DuckDBError) {
        std::string message = duckdb_prepare_error(stmt);
        duckdb_destroy_prepare(&stmt);
        throw DuckyError(message);
    }
    return new PreparedStatement(stmt, handle_);
}

std::shared_ptr<Result> Connection::current_result() {
    // Co-owning shared_ptr keeps the Result alive for the caller even if a
    // concurrent execute() replaces last_result_.
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

// Implemented here (rather than prepared.cpp) to reuse bind_parameters().

PreparedStatement::PreparedStatement(duckdb_prepared_statement stmt,
                                     std::shared_ptr<DuckDBHandle> handle)
    : stmt_(stmt), handle_(std::move(handle)) {}

PreparedStatement::~PreparedStatement() {
    if (stmt_) duckdb_destroy_prepare(&stmt_);
}

void PreparedStatement::ensure_valid() const {
    if (!stmt_) throw DuckyError("ducky: prepared statement is closed");
}

std::shared_ptr<Result> PreparedStatement::execute(nb::object parameters, bool streaming) {
    ensure_valid();
    if (!parameters.is_none()) {
        // Clear any prior binding so a positional set doesn't linger across calls.
        if (duckdb_clear_bindings(stmt_) == DuckDBError) {
            throw DuckyError("ducky: failed to clear prepared-statement bindings");
        }
        bind_parameters(stmt_, parameters);
    }
    duckdb_result result = run_pending(handle_->connection, stmt_, streaming);
    return std::make_shared<Result>(result, handle_);
}

void PreparedStatement::executemany(nb::object seq) {
    ensure_valid();
    for (nb::handle params : seq) {
        if (duckdb_clear_bindings(stmt_) == DuckDBError) {
            throw DuckyError("ducky: failed to clear prepared-statement bindings");
        }
        bind_parameters(stmt_, params);
        // Materialized — we discard the result, so streaming buys nothing here.
        duckdb_result result = run_pending(handle_->connection, stmt_, /*streaming=*/false);
        duckdb_destroy_result(&result);
    }
}

int64_t PreparedStatement::num_parameters() const {
    ensure_valid();
    return (int64_t)duckdb_nparams(stmt_);
}

nb::object PreparedStatement::parameter_name(int64_t index) const {
    ensure_valid();
    const char* name = duckdb_parameter_name(stmt_, (idx_t)index);
    if (!name) return nb::none();
    nb::object out = nb::str(name);
    duckdb_free((void*)name);
    return out;
}

std::vector<std::string> PreparedStatement::column_names() const {
    ensure_valid();
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
    ensure_valid();
    idx_t n = duckdb_prepared_statement_column_count(stmt_);
    std::vector<std::string> out;
    out.reserve(n);
    for (idx_t i = 0; i < n; ++i) {
        out.emplace_back(duckdb_type_name(duckdb_prepared_statement_column_type(stmt_, i)));
    }
    return out;
}

std::string PreparedStatement::statement_type() const {
    ensure_valid();
    return duckdb_statement_type_name(duckdb_prepared_statement_type(stmt_));
}

void PreparedStatement::close() {
    if (stmt_) duckdb_destroy_prepare(&stmt_);
    stmt_ = nullptr;
    handle_.reset();
}
