#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <tsl/robin_map.h>

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "database.hpp"
#include "duckdb.h"

namespace nb = nanobind;

class Connection;

// Buffered row, column, and Arrow appender.
class Appender {
   public:
    Appender(Connection& connection, std::string table, std::optional<std::string> schema,
             std::optional<std::string> catalog);
    ~Appender();

    Appender(const Appender&) = delete;
    Appender& operator=(const Appender&) = delete;

    // Target column names and types.
    std::vector<std::string> column_names() const;
    std::vector<std::string> column_types() const;

    // Append one row; ducky.DEFAULT selects the column default.
    void append_row(nb::args values);

    // Append named arrays with optional validity masks. Columns left out take
    // their DEFAULT, or NULL.
    void append_columns(nb::dict columns, nb::object masks);

    // Append a positionally and type-compatible Arrow stream or array.
    void append_arrow(nb::object source);

    // Flush without closing.
    void flush();

    // Flush and tear down. Idempotent.
    void close();

   private:
    // Locks the appender and raises if it is closed.
    std::unique_lock<std::mutex> lock_open();
    // Appends `n_rows` rows of the present columns through one reusable chunk.
    void append_column_slices(const std::vector<idx_t>& present,
                              const std::vector<nb::ndarray<nb::ro, nb::c_contig>>& arrays,
                              const std::vector<nb::ndarray<nb::ro, nb::c_contig>>& masks,
                              const std::vector<bool>& mask_present, int64_t n_rows);
    void check_state(duckdb_state state, const char* what);

    // Resolve a column name or throw.
    idx_t resolve(const std::string& name) const;

    duckdb_appender handle_ = nullptr;
    std::vector<std::string> names_;
    // Name to index in the schema vectors.
    tsl::robin_map<std::string, idx_t> name_to_idx_;
    std::vector<duckdb_logical_type> types_;  // owned; destroyed in dtor
    std::vector<duckdb_type> type_ids_;
    // Whether each column has a default expression.
    std::vector<bool> has_default_;
    std::shared_ptr<DuckDBHandle> connection_;
    // Guards handle_. Appends and flushes run with the GIL released, so
    // nb::lock_self alone doesn't keep close() out of them.
    std::mutex mu_;
};
