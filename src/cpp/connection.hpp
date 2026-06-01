#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <string>

#include "database.hpp"
#include "duckdb.h"
#include "result.hpp"

namespace nb = nanobind;

class PreparedStatement;

// A DuckDB database handle plus a single connection to it. Each Connection owns
// its own database, so connecting to ":memory:" yields an isolated in-memory
// database, matching duckdb.connect() semantics.
class Connection {
   public:
    // `config` is an optional dict of {name: string-value} pairs applied via
    // duckdb_set_config before opening the database. Invalid keys / values
    // raise DuckyError with DuckDB's error message.
    Connection(const std::string& database, nb::object config);
    ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    // Runs a query (optionally with positional parameters) and stashes the
    // result for the PEP 249-style fetch* methods. Returns *this so callers can
    // chain, e.g. con.execute(...).fetchall(). Pass `streaming=true` to receive
    // a lazy streaming result whose chunks are produced on demand (peak memory
    // bounded to one chunk; consumed once via the fetch / iter_batches path).
    Connection& execute(const std::string& query, nb::object parameters, bool streaming);

    // Eagerly runs a query and hands back its Result directly. See execute()
    // for the `streaming` semantics.
    std::shared_ptr<Result> sql(const std::string& query, bool streaming);

    // Compile `query` once into a PreparedStatement that can be executed
    // repeatedly with different parameters. Raises DuckyError on a parse/bind
    // failure. The statement shares this connection's DuckDBHandle.
    PreparedStatement* prepare(const std::string& query);

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

    // Best-effort interrupt of any query currently running on this
    // connection. Thread-safe: callable from a thread other than the one
    // blocked in execute(). No-op if no query is in flight.
    void interrupt();

    // Snapshot of the current query's progress. Returns (percentage,
    // rows_processed, total_rows_to_process). `percentage` is -1.0 when
    // DuckDB hasn't computed an estimate yet. Thread-safe.
    nb::tuple progress() const;

    // Register a Python object exposing the Arrow PyCapsule interface
    // (`__arrow_c_stream__`) as a table named `name`. The data is materialized
    // into a real DuckDB table at registration so subsequent queries against
    // `name` see the full dataset (the Arrow C stream is single-pass — a
    // lazy view would be drained by the first SELECT).
    void register_arrow(const std::string& name, nb::object obj);

    void close();

   private:
    void ensure_open() const;
    Result& current();
    duckdb_result run(const std::string& query, nb::object parameters, bool streaming);
    // Wraps a result, sharing ownership of the database/connection handle so the
    // result (and any Arrow stream from it) stays usable after this Connection
    // is closed or dropped.
    std::shared_ptr<Result> make_result(duckdb_result result);

    std::shared_ptr<DuckDBHandle> handle_;
    std::shared_ptr<Result> last_result_;
};
