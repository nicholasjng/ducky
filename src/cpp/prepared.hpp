#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "database.hpp"
#include "result.hpp"
#include "v2.hpp"

namespace nb = nanobind;

// Human-readable name for a statement type; unknown values fall back to
// "UNKNOWN".
inline const char* statement_type_name(DUCKDB_V2_STATEMENT_TYPE type) {
    switch (type) {
        case DUCKDB_V2_STATEMENT_TYPE_SELECT:
            return "SELECT";
        case DUCKDB_V2_STATEMENT_TYPE_INSERT:
            return "INSERT";
        case DUCKDB_V2_STATEMENT_TYPE_UPDATE:
            return "UPDATE";
        case DUCKDB_V2_STATEMENT_TYPE_CREATE:
            return "CREATE";
        case DUCKDB_V2_STATEMENT_TYPE_DELETE:
            return "DELETE";
        case DUCKDB_V2_STATEMENT_TYPE_PREPARE:
            return "PREPARE";
        case DUCKDB_V2_STATEMENT_TYPE_EXECUTE:
            return "EXECUTE";
        case DUCKDB_V2_STATEMENT_TYPE_ALTER:
            return "ALTER";
        case DUCKDB_V2_STATEMENT_TYPE_TRANSACTION:
            return "TRANSACTION";
        case DUCKDB_V2_STATEMENT_TYPE_COPY:
            return "COPY";
        case DUCKDB_V2_STATEMENT_TYPE_ANALYZE:
            return "ANALYZE";
        case DUCKDB_V2_STATEMENT_TYPE_VARIABLE_SET:
            return "VARIABLE_SET";
        case DUCKDB_V2_STATEMENT_TYPE_CREATE_FUNC:
            return "CREATE_FUNC";
        case DUCKDB_V2_STATEMENT_TYPE_EXPLAIN:
            return "EXPLAIN";
        case DUCKDB_V2_STATEMENT_TYPE_DROP:
            return "DROP";
        case DUCKDB_V2_STATEMENT_TYPE_EXPORT:
            return "EXPORT";
        case DUCKDB_V2_STATEMENT_TYPE_PRAGMA:
            return "PRAGMA";
        case DUCKDB_V2_STATEMENT_TYPE_VACUUM:
            return "VACUUM";
        case DUCKDB_V2_STATEMENT_TYPE_CALL:
            return "CALL";
        case DUCKDB_V2_STATEMENT_TYPE_SET:
            return "SET";
        case DUCKDB_V2_STATEMENT_TYPE_LOAD:
            return "LOAD";
        case DUCKDB_V2_STATEMENT_TYPE_RELATION:
            return "RELATION";
        case DUCKDB_V2_STATEMENT_TYPE_EXTENSION:
            return "EXTENSION";
        case DUCKDB_V2_STATEMENT_TYPE_LOGICAL_PLAN:
            return "LOGICAL_PLAN";
        case DUCKDB_V2_STATEMENT_TYPE_ATTACH:
            return "ATTACH";
        case DUCKDB_V2_STATEMENT_TYPE_DETACH:
            return "DETACH";
        case DUCKDB_V2_STATEMENT_TYPE_MULTI:
            return "MULTI";
        case DUCKDB_V2_STATEMENT_TYPE_COPY_DATABASE:
            return "COPY_DATABASE";
        case DUCKDB_V2_STATEMENT_TYPE_UPDATE_EXTENSIONS:
            return "UPDATE_EXTENSIONS";
        case DUCKDB_V2_STATEMENT_TYPE_MERGE_INTO:
            return "MERGE_INTO";
        case DUCKDB_V2_STATEMENT_TYPE_CONNECT:
            return "CONNECT";
        case DUCKDB_V2_STATEMENT_TYPE_DISCONNECT:
            return "DISCONNECT";
        case DUCKDB_V2_STATEMENT_TYPE_EXTERNAL_RESOURCE:
            return "EXTERNAL_RESOURCE";
        case DUCKDB_V2_STATEMENT_TYPE_PASSTHROUGH:
            return "PASSTHROUGH";
        default:
            return "UNKNOWN";
    }
}

// A C API v2 prepared statement of one SQL statement, with the parameter
// types its plan assumed.
class V2Prepared {
   public:
    // Parses and prepares `query`, which must hold exactly one statement.
    // Raises DuckyError with DuckDB's message on failure.
    V2Prepared(duckdb_v2_connection_handle con, const std::string& query);
    ~V2Prepared();

    V2Prepared(const V2Prepared&) = delete;
    V2Prepared& operator=(const V2Prepared&) = delete;

    duckdb_v2_prepared_statement_handle get() const { return prepared_; }
    // The type of the statement as written.
    DUCKDB_V2_STATEMENT_TYPE type() const { return type_; }
    // Binding keys in binding order: "1", "2", ... for positional
    // parameters, the name for named ones.
    const std::vector<std::string>& parameter_names() const { return parameter_names_; }
    // The position of parameter `key` in parameter_names(), or -1.
    int64_t parameter_index(const std::string& key) const;
    // The type the plan assumed for parameter `key`, or INVALID when it is
    // unresolved. A value of that type reuses the plan.
    DUCKDB_V2_LOGICAL_TYPE_ID planned_type(const std::string& key) const;
    // The statement's result columns as (name, type name) pairs, or none for
    // a statement that expands into several. Binds without the GIL.
    std::vector<std::pair<std::string, std::string>> columns(duckdb_v2_connection_handle con) const;

   private:
    duckdb_v2_sql_statement_handle sql_ = nullptr;
    duckdb_v2_prepared_statement_handle prepared_ = nullptr;
    DUCKDB_V2_STATEMENT_TYPE type_ = DUCKDB_V2_STATEMENT_TYPE_INVALID;
    std::vector<std::string> parameter_names_;
    std::vector<DUCKDB_V2_LOGICAL_TYPE_ID> parameter_types_;
};

// A reusable prepared statement sharing its connection handle.
class PreparedStatement {
   public:
    // `stmt` was prepared from `query` at Arrow registration `arrow_generation`.
    PreparedStatement(std::unique_ptr<V2Prepared> stmt, std::shared_ptr<DuckDBHandle> handle,
                      std::string query, uint64_t arrow_generation);
    ~PreparedStatement();

    PreparedStatement(const PreparedStatement&) = delete;
    PreparedStatement& operator=(const PreparedStatement&) = delete;

    // Bind parameters and execute, optionally streaming.
    std::shared_ptr<Result> execute(nb::object parameters, bool streaming);

    // Run the statement once per parameter set in `seq`, discarding each
    // result. The fast path for batched INSERT/UPDATE/DELETE.
    void executemany(nb::object seq);

    // Number of bind parameters in the statement.
    int64_t num_parameters() const;
    // Name of the parameter at `index` (1-based, matching DuckDB), or None if
    // the index is out of range. A positional parameter's name is its index.
    nb::object parameter_name(int64_t index) const;

    // Result schema known ahead of execution (empty for a statement that
    // expands into several).
    std::vector<std::string> column_names() const;
    std::vector<std::string> column_types() const;

    // The kind of statement (SELECT / INSERT / ...); see statement_type_name.
    std::string statement_type() const;

    void close();

   private:
    // Locks the statement and raises if it is closed. Prepares the statement
    // again if an Arrow source was registered since, which may have changed
    // the schema its plan read.
    std::unique_lock<std::mutex> lock_valid() const;

    // Replaced when the statement prepares again; still logically const.
    mutable std::unique_ptr<V2Prepared> stmt_;
    std::shared_ptr<DuckDBHandle> handle_;
    std::string query_;
    mutable uint64_t arrow_generation_ = 0;
    // Guards stmt_. Executions run with the GIL released, so nb::lock_self
    // alone doesn't keep close() or a re-prepare out of them.
    mutable std::mutex mu_;
};
