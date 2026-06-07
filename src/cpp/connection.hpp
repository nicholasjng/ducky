#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "database.hpp"
#include "duckdb.h"
#include "result.hpp"

namespace nb = nanobind;

class PreparedStatement;
struct ArrowRegistry;

// A steppable handle over a duckdb_pending_result — the substrate for an async
// execute. Where the synchronous `run_pending` drives the executor to
// completion in one C++ loop, this lets a Python coroutine own the loop:
// `execute_task()` advances one task (GIL released), and the coroutine decides
// what to do between ticks (yield to the event loop, check for cancellation).
//
// Concurrency: a tick offloaded via asyncio.to_thread keeps running on its
// worker thread even after the awaiting task is cancelled, so `drain()` (the
// cancellation teardown) must not race it. Every method that touches the
// pending result holds `mu_`; `drain()` first calls duckdb_interrupt (safe from
// another thread, and what makes the in-flight tick return promptly) and only
// then takes the lock. All blocking sections release the GIL so the worker
// thread can finish its tick and let the lock go.
class PendingResult {
   public:
    // Takes ownership of `pending`. If `owns_stmt`, also owns `stmt` and
    // destroys it once the result is materialized / drained (the Connection
    // path prepares a throwaway statement); the PreparedStatement path passes
    // owns_stmt=false since the statement outlives the pending result.
    PendingResult(duckdb_pending_result pending, duckdb_prepared_statement stmt, bool owns_stmt,
                  duckdb_connection con, std::shared_ptr<DuckDBHandle> handle);
    ~PendingResult();

    PendingResult(const PendingResult&) = delete;
    PendingResult& operator=(const PendingResult&) = delete;

    // Advance the executor by one task and return the resulting state
    // (READY / NOT_READY / NO_TASKS_AVAILABLE / ERROR). Releases the GIL.
    duckdb_pending_state execute_task();
    // DuckDB's message for the most recent error, or "" if the handle is gone.
    std::string error();
    // Materialize the finished result (call once execute_task reports READY).
    // Consumes the pending result; raises DuckyError if the execution errored.
    std::shared_ptr<Result> materialize();
    // Cancellation teardown: interrupt the connection, run the executor to a
    // terminal state so background workers detach, and discard the result.
    void drain();

   private:
    void cleanup_locked();

    std::mutex mu_;
    duckdb_pending_result pending_ = nullptr;
    duckdb_prepared_statement stmt_ = nullptr;
    bool owns_stmt_ = false;
    duckdb_connection con_ = nullptr;
    std::shared_ptr<DuckDBHandle> handle_;
};

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

    // Prepare + bind a query and hand back a steppable PendingResult *without*
    // driving the executor — the substrate for the async drive loop in
    // ducky._aio. The returned handle owns a throwaway prepared statement.
    PendingResult* make_pending(const std::string& query, nb::object parameters, bool streaming);

    // Compile `query` once into a PreparedStatement that can be executed
    // repeatedly with different parameters. Raises DuckyError on a parse/bind
    // failure. The statement shares this connection's DuckDBHandle.
    PreparedStatement* prepare(const std::string& query);

    nb::object fetchone();
    nb::list fetchmany(int64_t size);
    nb::list fetchall();
    nb::object fetchitem();
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

    // Post-execution profiling tree of the most recently run query, as a
    // nested {"metrics": {str: str}, "children": [...]} dict. Returns None if
    // profiling isn't enabled (`SET enable_profiling=...`). Metric values are
    // currently always strings (per the DuckDB C API); callers coerce.
    nb::object get_profiling_info() const;

    // Register a Python object exposing the Arrow PyCapsule interface
    // (`__arrow_c_stream__`) as a table named `name`. The data is materialized
    // into a real DuckDB table at registration so subsequent queries against
    // `name` see the full dataset (the Arrow C stream is single-pass — a
    // lazy view would be drained by the first SELECT).
    void register_arrow(const std::string& name, nb::object obj);

    void close();

   private:
    void ensure_open() const;
    duckdb_result run(const std::string& query, nb::object parameters, bool streaming);
    // Wraps a result, sharing ownership of the database/connection handle so the
    // result (and any Arrow stream from it) stays usable after this Connection
    // is closed or dropped.
    std::shared_ptr<Result> make_result(duckdb_result result);

    std::shared_ptr<DuckDBHandle> handle_;
    std::shared_ptr<Result> last_result_;
    // Lazily created on the first register_arrow; backs the Arrow replacement
    // scan. Declared last so it's torn down first (its nb::object refs decref
    // before the database handle closes). ~Connection is defined out-of-line in
    // connection.cpp, where ArrowRegistry is a complete type.
    std::unique_ptr<ArrowRegistry> arrow_registry_;
};
