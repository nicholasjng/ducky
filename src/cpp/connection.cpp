#include "connection.hpp"

#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "ducky.hpp"

namespace {

// Binds a Python sequence of positional parameters onto a prepared statement.
void bind_parameters(duckdb_prepared_statement stmt, nb::handle parameters) {
    idx_t param = 0;
    for (nb::handle value : parameters) {
        param += 1;  // DuckDB params are 1-indexed.
        duckdb_state state;
        if (value.is_none()) {
            state = duckdb_bind_null(stmt, param);
        } else if (nb::isinstance<nb::bool_>(value)) {
            state = duckdb_bind_boolean(stmt, param, nb::cast<bool>(value));
        } else if (nb::isinstance<nb::int_>(value)) {
            state = duckdb_bind_int64(stmt, param, nb::cast<int64_t>(value));
        } else if (nb::isinstance<nb::float_>(value)) {
            state = duckdb_bind_double(stmt, param, nb::cast<double>(value));
        } else if (nb::isinstance<nb::str>(value)) {
            state = duckdb_bind_varchar(stmt, param,
                                        nb::cast<std::string>(value).c_str());
        } else {
            throw DuckyError("ducky: unsupported parameter type at position " +
                             std::to_string(param));
        }
        if (state == DuckDBError) {
            throw DuckyError("ducky: failed to bind parameter " +
                             std::to_string(param));
        }
    }
}

}  // namespace

Connection::Connection(const std::string &database) {
    auto handle = std::make_shared<DuckDBHandle>();
    char *error = nullptr;
    if (duckdb_open_ext(database.c_str(), &handle->database, nullptr, &error) == DuckDBError) {
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

duckdb_result Connection::run(const std::string &query, nb::object parameters) {
    ensure_open();
    duckdb_result result;

    // Fast path: no parameters, run the query directly.
    if (parameters.is_none()) {
        if (duckdb_query(handle_->connection, query.c_str(), &result) == DuckDBError) {
            std::string message = duckdb_result_error(&result);
            duckdb_destroy_result(&result);
            throw DuckyError(message);
        }
        return result;
    }

    // Parameterized path: prepare, bind, execute.
    duckdb_prepared_statement stmt = nullptr;
    if (duckdb_prepare(handle_->connection, query.c_str(), &stmt) == DuckDBError) {
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
    duckdb_state state = duckdb_execute_prepared(stmt, &result);
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

Connection &Connection::execute(const std::string &query, nb::object parameters) {
    last_result_ = make_result(run(query, parameters));
    return *this;
}

std::shared_ptr<Result> Connection::sql(const std::string &query) {
    return make_result(run(query, nb::none()));
}

Result &Connection::current() {
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
