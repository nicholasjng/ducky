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

// Steppable pending result used by ducky._aio.
class PendingResult {
   public:
    // Takes `pending`, and `stmt` when `owns_stmt`.
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

    // Run a query and return its Result directly.
    std::shared_ptr<Result> sql(const std::string& query, bool streaming);

    // Run once per parameter set and discard the results.
    void executemany(const std::string& query, nb::object seq);

    // Execute with the Arrow collector and return a stream capsule.
    nb::object arrow_collected_stream(const std::string& query, nb::object parameters,
                                      uint64_t rows_per_batch);

    // Prepare + bind without driving the executor — substrate for ducky._aio.
    // The returned handle owns a throwaway prepared statement.
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
    duckdb_connection raw_connection() {
        ensure_open();
        return handle_->connection;
    }

    // Shared database and connection owner.
    std::shared_ptr<DuckDBHandle> handle() const { return handle_; }

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
    void ensure_open() const;
    duckdb_result run(const std::string& query, nb::object parameters, bool streaming);

    // Prepared-statement LRU keyed by exact query text.
    struct CachedStatement {
        std::string query;
        duckdb_prepared_statement stmt;
    };
    // Returns the cached statement for `query` (removing it from the cache),
    // or nullptr on a miss.
    duckdb_prepared_statement statement_cache_checkout(const std::string& query);
    // Returns a statement to the cache as most-recently-used, evicting the
    // least-recently-used entry past capacity.
    void statement_cache_checkin(const std::string& query, duckdb_prepared_statement stmt);
    // Destroys every cached statement. Must run before the connection closes.
    void statement_cache_clear();
    // Post-run disposal of a checked-out statement: flushes plans its execution
    // invalidated, then caches or destroys it.
    void statement_cache_retire(const std::string& query, duckdb_prepared_statement stmt);

    // One execution's checked-out or freshly prepared statement.
    class StatementLease {
       public:
        // Check out an unbound cached statement or prepare one.
        StatementLease(Connection& con, const std::string& query);
        ~StatementLease();

        StatementLease(const StatementLease&) = delete;
        StatementLease& operator=(const StatementLease&) = delete;

        duckdb_prepared_statement get() const { return stmt_; }
        // Hands the statement back to the cache per statement_cache_retire. Call
        // only after the execution succeeded.
        void retire();

       private:
        Connection& con_;
        const std::string& query_;
        duckdb_prepared_statement stmt_ = nullptr;
    };
    // Make a result sharing the database handle.
    std::shared_ptr<Result> make_result(duckdb_result result);
    // Fire the sink for sampled materialized queries.
    void maybe_emit_profile(const std::string& query);

    // Front is most recently used.
    std::list<CachedStatement> statement_lru_;
    tsl::robin_map<std::string, std::list<CachedStatement>::iterator> statement_cache_;

    // Guards handle_ reads and reset; never held across DuckDB calls.
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
