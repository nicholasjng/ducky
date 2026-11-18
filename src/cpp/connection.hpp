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

// Steppable, so ducky._aio can interleave a query with the event loop.
class PendingResult {
   public:
    PendingResult(std::unique_ptr<Execution> execution, std::shared_ptr<DuckDBHandle> handle);
    ~PendingResult();

    PendingResult(const PendingResult&) = delete;
    PendingResult& operator=(const PendingResult&) = delete;

    // Releases the GIL, since workers may run Python callbacks.
    PendingState execute_task();
    // Waits while DuckDB's workers own the remaining tasks, rather than
    // backing off with sleeps.
    PendingState execute_tasks(double budget);
    std::string error();
    // Runs the query to READY first, so it is safe to call early.
    std::shared_ptr<Result> materialize();
    void drain();

   private:
    // Requires mu_ and no GIL.
    PendingState run_locked(std::chrono::steady_clock::time_point deadline);
    // Requires mu_ and no GIL.
    void cancel_locked();

    std::mutex mu_;
    std::unique_ptr<Execution> execution_;
    std::string error_;
    std::shared_ptr<DuckDBHandle> handle_;
};

class Connection {
   public:
    // `config` is an optional Mapping[str, str].
    Connection(const std::string& database, nb::object config);
    ~Connection();

    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    Connection& execute(const std::string& query, nb::object parameters, bool streaming);

    // Leaves the retained result in place, unlike execute().
    std::shared_ptr<Result> sql(const std::string& query, nb::object parameters, bool streaming);

    void executemany(const std::string& query, nb::object seq);

    nb::object arrow_collected_stream(const std::string& query, nb::object parameters,
                                      uint64_t rows_per_batch);

    PendingResult* make_pending(const std::string& query, nb::object parameters, bool streaming);

    PreparedStatement* prepare(const std::string& query);

    nb::object fetchone();
    nb::list fetchmany(int64_t size);
    nb::list fetchall();
    nb::object fetchitem();
    nb::object description() const;
    nb::object columns() const;

    std::shared_ptr<Result> current_result();

    duckdb_connection raw_connection() { return open_handle()->connection; }

    // nullptr after close(). Callers hold the reference for the whole call, so
    // a concurrent close() cannot tear the connection down underneath them.
    std::shared_ptr<DuckDBHandle> handle() const;
    std::shared_ptr<DuckDBHandle> open_handle() const;

    // Thread-safe, and a no-op without a query in flight.
    void interrupt();

    // `percentage` is -1.0 until DuckDB has an estimate. Thread-safe.
    nb::tuple progress() const;

    nb::object get_profiling_info() const;

    void set_profile_sink(nb::object sink, int64_t sample, const std::string& mode);

    void register_arrow(const std::string& name, nb::object obj);

    void load_extension(const std::string& name);

    void close();

    int gc_traverse(visitproc visit, void* arg) const;
    void gc_clear();

   private:
    std::shared_ptr<Result> run(const std::string& query, nb::object parameters, bool streaming);

    // Prepared-statement LRU keyed by exact query text.
    struct CachedStatement {
        std::string query;
        std::unique_ptr<V2Prepared> stmt;
    };
    // Removes the statement, so no other execution shares it.
    std::unique_ptr<V2Prepared> statement_cache_checkout(const std::string& query);
    void statement_cache_checkin(const std::string& query, std::unique_ptr<V2Prepared> stmt);
    // Must run before the connection closes.
    void statement_cache_clear();
    void statement_cache_retire(const std::string& query, std::unique_ptr<V2Prepared> stmt);

    class StatementLease {
       public:
        StatementLease(Connection& con, duckdb_v2_connection_handle connection,
                       const std::string& query);

        const V2Prepared& get() const { return *stmt_; }
        // Only after the execution succeeded, so a statement whose execution
        // threw is destroyed instead of cached.
        void retire();

       private:
        Connection& con_;
        const std::string& query_;
        std::unique_ptr<V2Prepared> stmt_;
    };
    void maybe_emit_profile(duckdb_v2_connection_handle connection, const std::string& query);

    // Front is most recently used.
    std::list<CachedStatement> statement_lru_;
    tsl::robin_map<std::string, std::list<CachedStatement>::iterator> statement_cache_;

    // Never held across DuckDB calls, so close() cannot wait on a query. Read
    // handle_ only through handle() and open_handle().
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
