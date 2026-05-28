#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <string>

#include "database.hpp"
#include "duckdb.h"
#include "result.hpp"

namespace nb = nanobind;

// A DuckDB database handle plus a single connection to it. Each Connection owns
// its own database, so connecting to ":memory:" yields an isolated in-memory
// database, matching duckdb.connect() semantics.
class Connection {
   public:
    explicit Connection(const std::string& database);
    ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    // Runs a query (optionally with positional parameters) and stashes the
    // result for the PEP 249-style fetch* methods. Returns *this so callers can
    // chain, e.g. con.execute(...).fetchall().
    Connection& execute(const std::string& query, nb::object parameters);

    // Eagerly runs a query and hands back its Result directly.
    std::shared_ptr<Result> sql(const std::string& query);

    nb::object fetchone();
    nb::list fetchmany(int64_t size);
    nb::list fetchall();
    nb::object description() const;
    nb::object columns() const;

    // The last result produced by execute(), for the DataFrame/Arrow accessors.
    std::shared_ptr<Result> current_result();

    // Raw C-API connection handle, valid until close(). Used by UDF
    // registration in function.cpp; not exposed to Python.
    duckdb_connection raw_connection() {
        ensure_open();
        return handle_->connection;
    }

    // Shared owner of the database + connection. Used by Appender so the
    // database stays open for as long as the appender is alive.
    std::shared_ptr<DuckDBHandle> handle() const { return handle_; }

    void close();

   private:
    void ensure_open() const;
    Result& current();
    duckdb_result run(const std::string& query, nb::object parameters);
    // Wraps a result, sharing ownership of the database/connection handle so the
    // result (and any Arrow stream from it) stays usable after this Connection
    // is closed or dropped.
    std::shared_ptr<Result> make_result(duckdb_result result);

    std::shared_ptr<DuckDBHandle> handle_;
    std::shared_ptr<Result> last_result_;
};
