#pragma once

#include <nanobind/nanobind.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "database.hpp"
#include "duckdb.h"

namespace nb = nanobind;

class Connection;

// Wraps a `duckdb_appender` — the fastest bulk-insert path in DuckDB. Two
// modes:
//
//   append_row(*values)
//       Row-at-a-time. Dispatches each Python value onto the matching
//       duckdb_append_* C call based on the target column's logical type.
//       Covers VARCHAR / BLOB and the temporal types out of the box; the
//       intended path for small / mixed writes.
//
//   append_columns({name: ndarray, ...}, masks={name: bool ndarray, ...})
//       Columnar fast path. Builds duckdb_data_chunk(s) of up to
//       STANDARD_VECTOR_SIZE rows and hands them to duckdb_append_data_chunk.
//       v1 covers numeric / temporal dtypes (the same set Chunk.column
//       exposes on the read side). VARCHAR / nested types must use the row
//       API; an Arrow ingest variant (via __arrow_c_array__) is on the
//       roadmap and will subsume the gap.
//
// Lifetime: the Appender holds a shared_ptr to the connection's DuckDBHandle,
// so the database stays open as long as the appender is alive. Destruction
// calls duckdb_appender_destroy; pending rows are flushed on close() or on
// context-manager exit. Letting an Appender be garbage-collected with
// unflushed rows is best-effort — call close() / use `with` for durability.
class Appender {
   public:
    Appender(Connection& connection, std::string table, std::optional<std::string> schema,
             std::optional<std::string> catalog);
    ~Appender();

    Appender(const Appender&) = delete;
    Appender& operator=(const Appender&) = delete;

    // Number of target columns, and their DuckDB type names. Useful for
    // tests and error messages; mirrors Result.columns / Result.types.
    std::vector<std::string> column_names() const;
    std::vector<std::string> column_types() const;

    // Append one row's worth of values. `values` length must match the
    // target column count; types are coerced per-column against the cached
    // logical types.
    void append_row(nb::args values);

    // Append a batch of columns. Keys must be a subset of column_names();
    // any missing column is filled with NULL. `masks`, if provided, maps
    // column name -> 1-D bool/uint8 ndarray of the same length as that
    // column (True / non-zero = valid).
    void append_columns(nb::dict columns, nb::object masks);

    // Push any pending rows to the underlying table without invalidating
    // the appender. Errors raise DuckyError.
    void flush();

    // Flush and tear down. Idempotent.
    void close();

   private:
    void ensure_open() const;
    void check_state(duckdb_state state, const char* what);

    // Resolve a column name to its index in the appender's schema. Raises
    // DuckyError if the name is unknown.
    idx_t resolve(const std::string& name) const;

    duckdb_appender handle_ = nullptr;
    std::vector<std::string> names_;
    std::vector<duckdb_logical_type> types_;  // owned; destroyed in dtor
    std::vector<duckdb_type> type_ids_;
    std::shared_ptr<DuckDBHandle> connection_;
};
