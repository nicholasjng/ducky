#include "connection.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include "ducky.hpp"
#include "prepared.hpp"

namespace {

using namespace nb::literals;

// Cached Python stdlib type objects we dispatch against during binding. Same
// pattern as result.cpp's py_types() — imported once, intentionally leaked so
// the destructors never run at interpreter shutdown.
struct BindTypes {
    nb::type_object date;             // datetime.date
    nb::type_object time;             // datetime.time
    nb::type_object datetime;         // datetime.datetime
    nb::type_object timedelta;        // datetime.timedelta
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
        nb::module_ dt = nb::module_::import_("datetime");
        nb::type_object datetime_ty = dt.attr("datetime");
        nb::object utc = dt.attr("timezone").attr("utc");
        nb::object epoch_utc = datetime_ty(1970, 1, 1, 0, 0, 0, 0, utc);
        nb::object two_pow_127 = nb::int_(1).attr("__lshift__")(127);
        nb::object int128_min = two_pow_127.attr("__neg__")();
        nb::object two_pow_128 = nb::int_(1).attr("__lshift__")(128);
        nb::object mask64 = nb::int_(1).attr("__lshift__")(64).attr("__sub__")(nb::int_(1));
        return BindTypes{
            dt.attr("date"),
            dt.attr("time"),
            datetime_ty,
            dt.attr("timedelta"),
            nb::module_::import_("decimal").attr("Decimal"),
            nb::module_::import_("uuid").attr("UUID"),
            std::move(epoch_utc),
            std::move(int128_min),
            std::move(two_pow_127),
            std::move(two_pow_128),
            std::move(mask64),
        };
    });
}

// Split a Python int into a 128-bit (lo, hi) pair.
//
// `is_signed=true`  accepts values in [-2^127,   2^127 - 1].
// `is_signed=false` accepts values in [0,        2^128 - 1].
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

    // Two's-complement of N bits is (value mod 2^N); Python's % on negatives
    // already returns the positive residue.
    nb::object wrapped = nb::borrow(value).attr("__mod__")(T.uint128_max_plus_one);
    lo = nb::cast<uint64_t>(wrapped.attr("__and__")(T.mask64));
    hi = nb::cast<uint64_t>(wrapped.attr("__rshift__")(nb::int_(64)));
}

// Convert a tz-aware datetime to UTC microseconds since the unix epoch.
int64_t aware_datetime_to_utc_micros(nb::handle value) {
    nb::object delta = nb::borrow(value) - bind_types().epoch_utc;  // datetime.timedelta
    int64_t days = nb::cast<int64_t>(delta.attr("days"));
    int64_t seconds = nb::cast<int64_t>(delta.attr("seconds"));
    int64_t micros = nb::cast<int64_t>(delta.attr("microseconds"));
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
        // Try int64 first; on overflow, fall back to (u)hugeint.
        try {
            int64_t v = nb::cast<int64_t>(value);
            bind_check(duckdb_bind_int64(stmt, idx, v), idx);
            return;
        } catch (const nb::cast_error&) {
            // fallthrough
        }
        bool negative = PyObject_RichCompareBool(value.ptr(), nb::int_(0).ptr(), Py_LT) == 1;
        uint64_t lo = 0, hi = 0;
        if (negative) {
            py_long_to_int128(value, /*is_signed=*/true, lo, hi);
            duckdb_hugeint h{lo, static_cast<int64_t>(hi)};
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
        std::string s = nb::cast<std::string>(value);
        bind_check(duckdb_bind_varchar_length(stmt, idx, s.data(), s.size()), idx);
        return;
    }
    if (nb::isinstance<nb::bytes>(value)) {
        nb::bytes b = nb::borrow<nb::bytes>(value);
        bind_check(duckdb_bind_blob(stmt, idx, b.c_str(), b.size()), idx);
        return;
    }
    // datetime.datetime is a subclass of datetime.date, so check it first.
    if (nb::isinstance(value, T.datetime)) {
        bool aware = !value.attr("tzinfo").is_none();
        if (aware) {
            duckdb_timestamp ts{aware_datetime_to_utc_micros(value)};
            bind_check(duckdb_bind_timestamp_tz(stmt, idx, ts), idx);
        } else {
            duckdb_timestamp_struct s{};
            s.date.year = nb::cast<int32_t>(value.attr("year"));
            s.date.month = nb::cast<int8_t>(value.attr("month"));
            s.date.day = nb::cast<int8_t>(value.attr("day"));
            s.time.hour = nb::cast<int8_t>(value.attr("hour"));
            s.time.min = nb::cast<int8_t>(value.attr("minute"));
            s.time.sec = nb::cast<int8_t>(value.attr("second"));
            s.time.micros = nb::cast<int32_t>(value.attr("microsecond"));
            bind_check(duckdb_bind_timestamp(stmt, idx, duckdb_to_timestamp(s)), idx);
        }
        return;
    }
    if (nb::isinstance(value, T.date)) {
        duckdb_date_struct s{};
        s.year = nb::cast<int32_t>(value.attr("year"));
        s.month = nb::cast<int8_t>(value.attr("month"));
        s.day = nb::cast<int8_t>(value.attr("day"));
        bind_check(duckdb_bind_date(stmt, idx, duckdb_to_date(s)), idx);
        return;
    }
    if (nb::isinstance(value, T.time)) {
        duckdb_time_struct s{};
        s.hour = nb::cast<int8_t>(value.attr("hour"));
        s.min = nb::cast<int8_t>(value.attr("minute"));
        s.sec = nb::cast<int8_t>(value.attr("second"));
        s.micros = nb::cast<int32_t>(value.attr("microsecond"));
        bind_check(duckdb_bind_time(stmt, idx, duckdb_to_time(s)), idx);
        return;
    }
    if (nb::isinstance(value, T.timedelta)) {
        duckdb_interval iv{};
        iv.months = 0;
        iv.days = nb::cast<int32_t>(value.attr("days"));
        int64_t seconds = nb::cast<int64_t>(value.attr("seconds"));
        int32_t micros = nb::cast<int32_t>(value.attr("microseconds"));
        iv.micros = seconds * 1000000 + micros;
        bind_check(duckdb_bind_interval(stmt, idx, iv), idx);
        return;
    }
    if (nb::isinstance(value, T.decimal) || nb::isinstance(value, T.uuid)) {
        // Pragmatic v1: bind as VARCHAR; DuckDB implicitly casts to the
        // target column type (DECIMAL / UUID) on execute.
        nb::str s = nb::str(value);
        bind_check(duckdb_bind_varchar(stmt, idx, s.c_str()), idx);
        return;
    }
    // numpy scalars (and any object exposing .item() -> native Python): re-
    // dispatch on the unboxed value. We probe with hasattr to avoid pulling
    // numpy in as a runtime dependency.
    if (nb::hasattr(value, "item")) {
        nb::object item;
        try {
            item = value.attr("item")();
        } catch (...) {
            // .item() exists but failed (e.g. it's a method on a non-scalar
            // ndarray); fall through to the unsupported-type error.
            item = nb::object();
        }
        if (item) {
            // Guard against infinite recursion: the unboxed value must be a
            // *different* type than the original. Native Python scalars never
            // expose .item(), so this terminates immediately.
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

// Bind a sequence of positional parameters (list/tuple) onto a prepared
// statement.
void bind_positional(duckdb_prepared_statement stmt, nb::handle parameters) {
    idx_t param = 0;
    for (nb::handle value : parameters) {
        param += 1;  // DuckDB params are 1-indexed.
        bind_one(stmt, param, value);
    }
}

// Bind a dict of named parameters by looking up each name's index.
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

// Drive duckdb_pending_execute_task in a GIL-releasing loop until the executor is
// READY (or errors), then materialize the duckdb_result via duckdb_execute_pending.
// `streaming` switches between the buffered result (duckdb_pending_prepared) and
// the lazy chunk-pull result (duckdb_pending_prepared_streaming) — the latter
// keeps peak memory bounded to one chunk for iter_batches_* consumers.
//
// Ticking one task at a time (instead of blocking inside duckdb_execute_prepared)
// gives two things the v1 gil_scoped_release path could not:
//   - KeyboardInterrupt lands mid-query: PyErr_CheckSignals runs between ticks,
//     so Ctrl-C triggers duckdb_interrupt + a drained teardown instead of
//     parking until the query finishes.
//   - Streaming results are reachable: the materialized fast path through
//     duckdb_execute_prepared has no streaming counterpart.
//
// The prepared statement is not destroyed here — the caller still owns it.
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
            // Workers own the remaining tasks; yield briefly so we don't spin
            // re-querying state. 1ms keeps Ctrl-C latency imperceptible.
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
            // A Python signal handler raised (typically KeyboardInterrupt).
            // Interrupt the executor, drain to a terminal state so background
            // workers detach cleanly, then re-raise. The drained result is
            // discarded — we never expose a partial result to the caller.
            duckdb_interrupt(con);
            duckdb_result drain;
            if (duckdb_execute_pending(pending, &drain) == DuckDBSuccess) {
                duckdb_destroy_result(&drain);
            } else {
                duckdb_destroy_result(&drain);
            }
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

    // Build a duckdb_config from the user-supplied Mapping[str, str], if any.
    // We iterate via .items() rather than requiring an nb::dict so the C
    // surface honestly accepts any read-only Mapping (TypedDict, MappingProxy,
    // user mappings) — we only read, never mutate or take ownership. The
    // config is always destroyed; DuckDB copies the settings on open.
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

void Connection::register_arrow(const std::string& name, nb::object obj) {
    ensure_open();
    // The name is interpolated into SQL below. Enforce a strict identifier
    // shape so we never need to quote-escape: [A-Za-z][A-Za-z0-9_]*.
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
    nb::object capsule = obj.attr("__arrow_c_stream__")();
    void* raw = PyCapsule_GetPointer(capsule.ptr(), "arrow_array_stream");
    if (!raw) {
        PyErr_Clear();
        throw DuckyError(
            "ducky: __arrow_c_stream__ did not return an 'arrow_array_stream' "
            "PyCapsule");
    }

    // The Arrow C stream is single-pass. Stage it as a temporary view, then
    // materialize into a real table so repeated queries against `name` see
    // the data. The capsule's destructor calls stream->release after we
    // return; arrow_scan has nullified `release` by then so it's a no-op.
    static std::atomic<uint64_t> counter{0};
    std::string staging = "__ducky_arrow_" + std::to_string(counter.fetch_add(1));

    duckdb_arrow_stream stream = (duckdb_arrow_stream)raw;
    if (duckdb_arrow_scan(handle_->connection, staging.c_str(), stream) == DuckDBError) {
        throw DuckyError("ducky: failed to stage Arrow stream");
    }

    // CREATE OR REPLACE drops any prior table with this name; the temp view
    // is dropped explicitly to keep the catalog clean even on error.
    std::string materialize = "CREATE OR REPLACE TABLE " + name + " AS SELECT * FROM " + staging;
    duckdb_result r;
    duckdb_state st = duckdb_query(handle_->connection, materialize.c_str(), &r);
    std::string err;
    if (st == DuckDBError) {
        err = duckdb_result_error(&r);
    }
    duckdb_destroy_result(&r);

    std::string drop = "DROP VIEW IF EXISTS " + staging;
    duckdb_result rd;
    duckdb_query(handle_->connection, drop.c_str(), &rd);
    duckdb_destroy_result(&rd);

    if (st == DuckDBError) {
        throw DuckyError("ducky: failed to materialize Arrow stream into '" + name + "': " + err);
    }
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

    // Unified prepare + pending-execute path. Going through prepare even when
    // there are no parameters costs one extra parse per call (negligible vs the
    // executor work) but buys us pending semantics — see run_pending() for the
    // GIL / signal-handling / streaming rationale. Multi-statement strings are
    // a deliberate non-feature here; the unbound `duckdb_extract_statements`
    // API is the roadmap path if we ever want them back.
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

// ── Async substrate: PendingResult + Connection::make_pending ───────────────

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
    // Backstop for an abandoned coroutine (never materialized or drained). A
    // live worker would hold a reference to this object via the bound
    // execute_task method, so no tick can be in flight here — no lock needed.
    // Interrupt first so a partially-started query stops promptly.
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
    // Interrupt is safe to call from this thread while a worker is mid-tick on
    // another; it makes that tick return promptly so the lock frees up. Only
    // then do we take the lock (with the GIL dropped, so the worker can finish
    // and reacquire the GIL to return) and run the executor to a terminal state.
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
    // Returning by value hands back a co-owning shared_ptr, so the Result stays
    // alive for the caller even if a concurrent execute() replaces last_result_
    // the instant after. The execute()/fetch*/current_result bindings carry
    // nb::lock_self() (see ducky.cpp), serializing those last_result_ accesses on
    // free-threaded builds.
    if (!last_result_) throw DuckyError("ducky: no result set; call execute() first");
    return last_result_;
}

// The fetch delegators go through current_result() (a co-owning shared_ptr)
// rather than a bare Result& — the temporary keeps the result alive across the
// call, and Result's own lock_self serializes the cursor mutation.
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

// ── PreparedStatement ────────────────────────────────────────────────────────
// Implemented here (rather than a separate prepared.cpp) to reuse the
// bind_parameters() machinery in this file's anonymous namespace.

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
