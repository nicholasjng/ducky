#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "database.hpp"
#include "duckdb.h"
#include "result.hpp"

namespace nb = nanobind;

// Human-readable name for a duckdb_statement_type. Mirrors the enum in
// duckdb.h; unknown values fall back to "UNKNOWN".
inline const char* duckdb_statement_type_name(duckdb_statement_type type) {
    switch (type) {
        case DUCKDB_STATEMENT_TYPE_SELECT:
            return "SELECT";
        case DUCKDB_STATEMENT_TYPE_INSERT:
            return "INSERT";
        case DUCKDB_STATEMENT_TYPE_UPDATE:
            return "UPDATE";
        case DUCKDB_STATEMENT_TYPE_EXPLAIN:
            return "EXPLAIN";
        case DUCKDB_STATEMENT_TYPE_DELETE:
            return "DELETE";
        case DUCKDB_STATEMENT_TYPE_PREPARE:
            return "PREPARE";
        case DUCKDB_STATEMENT_TYPE_CREATE:
            return "CREATE";
        case DUCKDB_STATEMENT_TYPE_EXECUTE:
            return "EXECUTE";
        case DUCKDB_STATEMENT_TYPE_ALTER:
            return "ALTER";
        case DUCKDB_STATEMENT_TYPE_TRANSACTION:
            return "TRANSACTION";
        case DUCKDB_STATEMENT_TYPE_COPY:
            return "COPY";
        case DUCKDB_STATEMENT_TYPE_ANALYZE:
            return "ANALYZE";
        case DUCKDB_STATEMENT_TYPE_VARIABLE_SET:
            return "VARIABLE_SET";
        case DUCKDB_STATEMENT_TYPE_CREATE_FUNC:
            return "CREATE_FUNC";
        case DUCKDB_STATEMENT_TYPE_DROP:
            return "DROP";
        case DUCKDB_STATEMENT_TYPE_EXPORT:
            return "EXPORT";
        case DUCKDB_STATEMENT_TYPE_PRAGMA:
            return "PRAGMA";
        case DUCKDB_STATEMENT_TYPE_VACUUM:
            return "VACUUM";
        case DUCKDB_STATEMENT_TYPE_CALL:
            return "CALL";
        case DUCKDB_STATEMENT_TYPE_SET:
            return "SET";
        case DUCKDB_STATEMENT_TYPE_LOAD:
            return "LOAD";
        case DUCKDB_STATEMENT_TYPE_RELATION:
            return "RELATION";
        case DUCKDB_STATEMENT_TYPE_EXTENSION:
            return "EXTENSION";
        case DUCKDB_STATEMENT_TYPE_LOGICAL_PLAN:
            return "LOGICAL_PLAN";
        case DUCKDB_STATEMENT_TYPE_ATTACH:
            return "ATTACH";
        case DUCKDB_STATEMENT_TYPE_DETACH:
            return "DETACH";
        case DUCKDB_STATEMENT_TYPE_MULTI:
            return "MULTI";
        default:
            return "UNKNOWN";
    }
}

// A compiled SQL statement that can be executed repeatedly with different
// parameters. Created via Connection.prepare(); shares the connection's
// DuckDBHandle so the database stays open for as long as the statement (and any
// Result it produced) is alive. The parse + bind + plan happens once at
// prepare() time, so executing in a loop avoids re-compiling the query.
class PreparedStatement {
   public:
    PreparedStatement(duckdb_prepared_statement stmt, std::shared_ptr<DuckDBHandle> handle);
    ~PreparedStatement();

    PreparedStatement(const PreparedStatement&) = delete;
    PreparedStatement& operator=(const PreparedStatement&) = delete;

    // Bind `parameters` (list/tuple positional, dict named, or None) and run
    // the statement, returning its Result. The executor is driven one task at
    // a time with the GIL released between ticks, so KeyboardInterrupt lands
    // mid-query and DuckDB's scheduler (plus any UDF workers) can run. Pass
    // `streaming=true` for a lazy chunk-pull result; see Connection::execute.
    std::shared_ptr<Result> execute(nb::object parameters, bool streaming);

    // Run the statement once per parameter set in `seq`, discarding each
    // result. The fast path for batched INSERT/UPDATE/DELETE.
    void executemany(nb::object seq);

    // Number of bind parameters in the statement.
    int64_t num_parameters() const;
    // Name of the parameter at `index` (1-based, matching DuckDB), or None if
    // the index is out of range or the parameter is positional/unnamed.
    nb::object parameter_name(int64_t index) const;

    // Result schema known ahead of execution (empty for non-result statements).
    std::vector<std::string> column_names() const;
    std::vector<std::string> column_types() const;

    // The kind of statement (SELECT / INSERT / ...); see
    // duckdb_statement_type_name.
    std::string statement_type() const;

    void close();

   private:
    void ensure_valid() const;

    duckdb_prepared_statement stmt_ = nullptr;
    std::shared_ptr<DuckDBHandle> handle_;
};
