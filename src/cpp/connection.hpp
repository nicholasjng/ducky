#pragma once

#include <nanobind/nanobind.h>
#include <tsl/robin_map.h>

#include <chrono>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>

#include "database.hpp"
#include "duckdb.h"
#include "prepared.hpp"
#include "result.hpp"

namespace nb = nanobind;

struct ArrowRegistry;

// The state of an async query after a tick. The values are C API v1's pending
// states, which ducky._aio was written against.
enum class PendingState : int { READY = 0, NOT_READY = 1, ERROR = 2, NO_TASKS = 3 };

// Steppable pending result used by ducky._aio.
class PendingResult {
   public:
    PendingResult(std::unique_ptr<Execution> execution, std::shared_ptr<DuckDBHandle> handle);
    ~PendingResult();

    PendingResult(const PendingResult&) = delete;
    PendingResult& operator=(const PendingResult&) = delete;

    // Runs one step of the query. Releases the GIL.
    PendingState execute_task();
    // Runs the query for up to `budget` seconds (GIL released), waiting while
    // DuckDB's workers own the remaining tasks.
    PendingState execute_tasks(double budget);
    // The message of the error execution failed with, or "".
    std::string error();
    // Returns the result once execute_task reports READY, running the query
    // to that point first. Raises DuckyError on execution failure.
    std::shared_ptr<Result> materialize();
    // Cancellation teardown: interrupt and end the query.
    void drain();

   private:
    // Requires mu_ and no GIL.
    PendingState run_locked(std::chrono::steady_clock::time_point deadline);
    // Interrupts and ends the query. Requires mu_ and no GIL.
    void cancel_locked();

    std::mutex mu_;
    std::unique_ptr<Execution> execution_;
    std::string error_;
    std::shared_ptr<DuckDBHandle> handle_;
};

// A DuckDB database and connection.
class Connection {
   public:
    // `config` is an optional Mapping[str, str].
    Connection(const std::string& database, nb::object config);
    ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    // Run a query and retain its result for fetch methods.
    Connection& execute(const std::string& query, nb::object parameters, bool streaming);

    // Run a query and return its Result directly, leaving the retained result
    // in place.
    std::shared_ptr<Result> sql(const std::string& query, nb::object parameters, bool streaming);

    // Run once per parameter set and discard the results.
    void executemany(const std::string& query, nb::object seq);

    // Execute as an Arrow result and return a stream capsule over its arrays.
    nb::object arrow_collected_stream(const std::string& query, nb::object parameters,
                                      uint64_t rows_per_batch);

    // Start the query without driving it — substrate for ducky._aio.
    PendingResult* make_pending(const std::string& query, nb::object parameters, bool streaming);

    // Compile a reusable prepared statement.
    PreparedStatement* prepare(const std::string& query);

    nb::object fetchone();
    nb::list fetchmany(int64_t size);
    nb::list fetchall();
    nb::object fetchitem();
    nb::object description() const;
    nb::object columns() const;

    // The last result from execute(), for the DataFrame/Arrow accessors.
    std::shared_ptr<Result> current_result();

    // Raw C API handle.
    duckdb_connection raw_connection() { return open_handle()->connection; }

    // Shared database and connection owner, or nullptr after close(). Callers
    // that run DuckDB work hold the returned reference for the whole call, so
    // a concurrent close() cannot tear the connection down underneath them.
    std::shared_ptr<DuckDBHandle> handle() const;
    // Same, but raises when the connection is closed.
    std::shared_ptr<DuckDBHandle> open_handle() const;

    // Thread-safe; no-op if no query is in flight.
    void interrupt();

    // Snapshot: (percentage, rows_processed, total_rows_to_process).
    // `percentage` is -1.0 until DuckDB has an estimate. Thread-safe.
    nb::tuple progress() const;

    // Return the profiling tree, or None when disabled.
    nb::object get_profiling_info() const;

    // Invoke a profile sink every `sample` materialized queries.
    void set_profile_sink(nb::object sink, int64_t sample, const std::string& mode);

    // Register a reusable Arrow stream source as a table.
    void register_arrow(const std::string& name, nb::object obj);

    // Load an extension by name or path.
    void load_extension(const std::string& name);

    void close();

    // Visit and clear Python references for cyclic GC.
    int gc_traverse(visitproc visit, void* arg) const;
    void gc_clear();

   private:
    // Runs `query` to a settled result.
    std::shared_ptr<Result> run(const std::string& query, nb::object parameters, bool streaming);

    // Prepared-statement LRU keyed by exact query text.
    struct CachedStatement {
        std::string query;
        std::unique_ptr<V2Prepared> stmt;
    };
    // Returns the cached statement for `query` (removing it from the cache),
    // or nullptr on a miss.
    std::unique_ptr<V2Prepared> statement_cache_checkout(const std::string& query);
    // Returns a statement to the cache as most-recently-used, evicting the
    // least-recently-used entry past capacity.
    void statement_cache_checkin(const std::string& query, std::unique_ptr<V2Prepared> stmt);
    // Destroys every cached statement. Must run before the connection closes.
    void statement_cache_clear();
    // Post-run disposal of a checked-out statement: flushes plans its execution
    // invalidated, then caches or destroys it.
    void statement_cache_retire(const std::string& query, std::unique_ptr<V2Prepared> stmt);

    // One execution's checked-out or freshly prepared statement.
    class StatementLease {
       public:
        // Check out a cached statement or prepare one on `connection`.
        StatementLease(Connection& con, duckdb_v2_connection_handle connection,
                       const std::string& query);

        const V2Prepared& get() const { return *stmt_; }
        // Hands the statement back to the cache per statement_cache_retire. Call
        // only after the execution succeeded; a statement whose execution threw
        // is destroyed instead.
        void retire();

       private:
        Connection& con_;
        const std::string& query_;
        std::unique_ptr<V2Prepared> stmt_;
    };
    // Fire the sink for sampled materialized queries.
    void maybe_emit_profile(duckdb_connection connection, const std::string& query);

    // Front is most recently used.
    std::list<CachedStatement> statement_lru_;
    tsl::robin_map<std::string, std::list<CachedStatement>::iterator> statement_cache_;

    // Guards handle_ reads and reset; never held across DuckDB calls. Read
    // handle_ only through handle() / open_handle().
    mutable nb::ft_mutex handle_mu_;
    std::shared_ptr<DuckDBHandle> handle_;
    std::shared_ptr<Result> last_result_;
    // Declared last so its Python references are released first.
    std::shared_ptr<ArrowRegistry> arrow_registry_;
    // nb::none() when no sink is installed.
    nb::object profile_sink_;
    int64_t profile_sample_ = 1;
    int64_t profile_counter_ = 0;
};
