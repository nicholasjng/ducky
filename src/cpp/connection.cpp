#include "connection.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <atomic>
#include <cstring>

#include "ducky.hpp"

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
    if (BindTypes* p = cached.load()) return *p;
    nb::module_ dt = nb::module_::import_("datetime");
    nb::type_object datetime_ty = dt.attr("datetime");
    nb::object utc = dt.attr("timezone").attr("utc");
    nb::object epoch_utc = datetime_ty(1970, 1, 1, 0, 0, 0, 0, utc);
    nb::object two_pow_127 = nb::int_(1).attr("__lshift__")(127);
    nb::object int128_min = two_pow_127.attr("__neg__")();
    nb::object two_pow_128 = nb::int_(1).attr("__lshift__")(128);
    nb::object mask64 = nb::int_(1).attr("__lshift__")(64).attr("__sub__")(nb::int_(1));
    auto box = new BindTypes{
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
    nb::ft_lock_guard lock(mu);
    if (BindTypes* p = cached.load()) return *p;
    cached.store(box);
    return *box;
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

duckdb_result Connection::run(const std::string& query, nb::object parameters) {
    ensure_open();
    duckdb_result result;
    duckdb_connection con = handle_->connection;

    // Fast path: no parameters, run the query directly.
    // We release the GIL across duckdb_query so DuckDB's parallel scheduler
    // can run worker threads — in particular, so workers can call back into
    // Python UDF trampolines (which re-acquire the GIL via nb::gil_scoped_acquire).
    // Without this release, the main thread holds the GIL while spinning in
    // Executor::ExecuteTask waiting for workers that themselves block on GIL acquisition,
    // a deterministic deadlock whenever DuckDB schedules the UDF on a worker.
    if (parameters.is_none()) {
        duckdb_state state;
        {
            nb::gil_scoped_release release;
            state = duckdb_query(con, query.c_str(), &result);
        }
        if (state == DuckDBError) {
            std::string message = duckdb_result_error(&result);
            duckdb_destroy_result(&result);
            throw DuckyError(message);
        }
        return result;
    }

    // Parameterized path: prepare, bind, execute. Prepare and bind keep the GIL
    // (they touch Python objects); only execute releases it.
    duckdb_prepared_statement stmt = nullptr;
    if (duckdb_prepare(con, query.c_str(), &stmt) == DuckDBError) {
        std::string message = duckdb_prepare_error(stmt);
        duckdb_destroy_prepare(&stmt);
        throw DuckyError(message);
    }
    try {
        bind_parameters(stmt, parameters);
    } catch (...) {
        duckdb_destroy_prepare(&stmt);
        throw;
    }
    duckdb_state state;
    {
        nb::gil_scoped_release release;
        state = duckdb_execute_prepared(stmt, &result);
    }
    duckdb_destroy_prepare(&stmt);
    if (state == DuckDBError) {
        std::string message = duckdb_result_error(&result);
        duckdb_destroy_result(&result);
        throw DuckyError(message);
    }
    return result;
}

std::shared_ptr<Result> Connection::make_result(duckdb_result result) {
    return std::make_shared<Result>(result, handle_);
}

Connection& Connection::execute(const std::string& query, nb::object parameters) {
    last_result_ = make_result(run(query, parameters));
    return *this;
}

std::shared_ptr<Result> Connection::sql(const std::string& query) {
    return make_result(run(query, nb::none()));
}

Result& Connection::current() {
    if (!last_result_) throw DuckyError("ducky: no result set; call execute() first");
    return *last_result_;
}

std::shared_ptr<Result> Connection::current_result() {
    if (!last_result_) throw DuckyError("ducky: no result set; call execute() first");
    return last_result_;
}

nb::object Connection::fetchone() { return current().fetchone(); }
nb::list Connection::fetchmany(int64_t size) { return current().fetchmany(size); }
nb::list Connection::fetchall() { return current().fetchall(); }

nb::object Connection::description() const {
    if (!last_result_) return nb::none();
    return last_result_->description();
}

nb::object Connection::columns() const {
    if (!last_result_) return nb::none();
    return nb::cast(last_result_->column_names());
}
