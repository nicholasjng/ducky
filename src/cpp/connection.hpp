#pragma once

#include <nanobind/nanobind.h>
#include <tsl/robin_map.h>

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>

#include "database.hpp"
#include "duckdb.h"
#include "result.hpp"

namespace nb = nanobind;

class PreparedStatement;
struct ArrowRegistry;

// A steppable handle over a duckdb_pending_result, driven from Python by
// ducky._aio. execute_task() advances one task (GIL released); the coroutine
// yields between ticks. drain() must not race an in-flight tick: it calls
// duckdb_interrupt first (safe cross-thread, ends the tick promptly) then
// takes mu_ with the GIL released so the worker can return.
class PendingResult {
   public:
    // Takes ownership of `pending`. If `owns_stmt`, also owns and destroys
    // `stmt` (the Connection path prepares a throwaway statement; the
    // PreparedStatement path passes owns_stmt=false).
    PendingResult(duckdb_pending_result pending, duckdb_prepared_statement stmt, bool owns_stmt,
                  duckdb_connection con, std::shared_ptr<DuckDBHandle> handle);
    ~PendingResult();

    PendingResult(const PendingResult&) = delete;
    PendingResult& operator=(const PendingResult&) = delete;

    // Advance the executor by one task. Releases the GIL.
    duckdb_pending_state execute_task();
    // DuckDB's most recent error message, or "" if the handle is gone.
    std::string error();
    // Materialize the finished result; call once execute_task reports READY.
    // Raises DuckyError on execution failure.
    std::shared_ptr<Result> materialize();
    // Cancellation teardown: interrupt, drain to a terminal state, discard.
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

// A DuckDB database handle plus one connection to it. Each Connection owns its
// own database, so two ":memory:" connections are isolated.
class Connection {
   public:
    // `config` is an optional Mapping[str, str] applied via duckdb_set_config
    // before open. Invalid keys/values raise DuckyError.
    Connection(const std::string& database, nb::object config);
    ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    // Run a query and stash the result for the PEP 249 fetch* methods. Returns
    // *this for chaining (e.g. con.execute(...).fetchall()).
    Connection& execute(const std::string& query, nb::object parameters, bool streaming);

    // Run a query and return its Result directly.
    std::shared_ptr<Result> sql(const std::string& query, bool streaming);

    // Prepare + bind without driving the executor — substrate for ducky._aio.
    // The returned handle owns a throwaway prepared statement.
    PendingResult* make_pending(const std::string& query, nb::object parameters, bool streaming);

    // Compile `query` once into a PreparedStatement reusable across executions
    // with different parameters.
    PreparedStatement* prepare(const std::string& query);

    nb::object fetchone();
    nb::list fetchmany(int64_t size);
    nb::list fetchall();
    nb::object fetchitem();
    nb::object description() const;
    nb::object columns() const;

    // The last result from execute(), for the DataFrame/Arrow accessors.
    std::shared_ptr<Result> current_result();

    // Raw C-API connection handle. Used by UDF registration in function.cpp.
    duckdb_connection raw_connection() {
        ensure_open();
        return handle_->connection;
    }

    // Shared owner of the database + connection. Used by Appender to keep the
    // database open for the appender's lifetime.
    std::shared_ptr<DuckDBHandle> handle() const { return handle_; }

    // Thread-safe; no-op if no query is in flight.
    void interrupt();

    // Snapshot: (percentage, rows_processed, total_rows_to_process).
    // `percentage` is -1.0 until DuckDB has an estimate. Thread-safe.
    nb::tuple progress() const;

    // Post-execution profiling tree as a nested
    // {"metrics": {str: str}, "children": [...]} dict, or None if profiling
    // isn't enabled. Values are strings (DuckDB C API); callers coerce.
    nb::object get_profiling_info() const;

    // Install an always-on profile sink: every execute()/sql() invokes
    // sink(query, info). Enables profiling on the connection. `sample=N` fires
    // every Nth query. Pass `sink=None` to detach; profiling settings are not
    // restored.
    void set_profile_sink(nb::object sink, int64_t sample, const std::string& mode);

    // Register an Arrow-PyCapsule source as a table named `name`. The source
    // is kept and re-streamed on each query via a replacement scan.
    void register_arrow(const std::string& name, nb::object obj);

    void close();

   private:
    void ensure_open() const;
    duckdb_result run(const std::string& query, nb::object parameters, bool streaming);

    // Prepared-statement cache for run(): an LRU keyed by exact query text so
    // repeated execute()/sql() calls skip parse + plan (parsing alone costs
    // tens of µs per query). Entries are *checked out* (removed) while a query
    // runs — run_pending releases the GIL between executor ticks, so another
    // thread could otherwise grab the same statement and clear/rebind it
    // mid-execution — and checked back in on success. Staleness is covered on
    // two fronts: catalog changes rebind cached plans inside DuckDB
    // (RebindPreparedStatement), and settings changes — which DuckDB does NOT
    // rebind on, despite folding e.g. current_setting() into plans — flush the
    // whole cache via keeps_stmt_cache() in connection.cpp.
    struct CachedStmt {
        std::string query;
        duckdb_prepared_statement stmt;
    };
    // Returns the cached statement for `query` (removing it from the cache),
    // or nullptr on a miss.
    duckdb_prepared_statement stmt_cache_checkout(const std::string& query);
    // Returns a statement to the cache as most-recently-used, evicting the
    // least-recently-used entry past capacity.
    void stmt_cache_checkin(const std::string& query, duckdb_prepared_statement stmt);
    // Destroys every cached statement. Must run before the connection closes.
    void stmt_cache_clear();
    // Shares ownership of the database/connection handle so the result (and
    // any Arrow stream from it) outlives this Connection.
    std::shared_ptr<Result> make_result(duckdb_result result);
    // Fire the sink if installed and the sample counter hits. Called from
    // run() for materialized queries only — streaming profiles would reflect
    // the previous query.
    void maybe_emit_profile(const std::string& query);

    // Front = most recently used. The map indexes list nodes by query text;
    // GIL builds serialize access via the GIL (the cache is only touched with
    // it held), free-threaded builds via nb::lock_self() on execute()/sql().
    std::list<CachedStmt> stmt_lru_;
    tsl::robin_map<std::string, std::list<CachedStmt>::iterator> stmt_cache_;

    std::shared_ptr<DuckDBHandle> handle_;
    std::shared_ptr<Result> last_result_;
    // Declared last so it tears down first: its nb::object refs decref before
    // the database handle closes. ~Connection is out-of-line so ArrowRegistry
    // can be incomplete here.
    std::unique_ptr<ArrowRegistry> arrow_registry_;
    // nb::none() when no sink is installed.
    nb::object profile_sink_;
    int64_t profile_sample_ = 1;
    int64_t profile_counter_ = 0;
};
