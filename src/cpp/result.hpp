#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "chunk.hpp"
#include "database.hpp"
#include "duckdb.h"

namespace nb = nanobind;

// A lazily decoded duckdb_result.
class Result {
   public:
    // `handle` keeps the database alive.
    Result(duckdb_result result, std::shared_ptr<DuckDBHandle> handle);
    ~Result();

    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;

    const std::vector<std::string>& column_names() const { return schema_->names; }
    std::vector<std::string> column_types() const;
    // PEP 249 Cursor.description: one 7-tuple per column, or None when the
    // statement produced no result set.
    nb::object description() const;

    // Returns the next row as a tuple, or None when the result is exhausted.
    nb::object fetchone();
    nb::list fetchmany(int64_t size);
    nb::list fetchall();
    // Returns the lone scalar of a 1-row × 1-column result. Raises if the result
    // does not have exactly one column or does not yield exactly one row.
    nb::object fetchitem();

    // Rows affected by an INSERT / UPDATE / DELETE, per duckdb_rows_changed.
    // 0 for statements that don't report changed rows (SELECT, DDL) and for
    // streaming results (the C API can't read the count without materializing).
    int64_t rows_changed();

    // Pull the next chunk. Shares the cursor with other consumers.
    nb::object fetch_chunk();

    // Pull the next chunk as NumPy arrays.
    nb::object fetch_numpy_chunk();

    // Drain selected flat columns into NumPy arrays.
    nb::object to_numpy(nb::object columns);

    // Export the remaining rows as an Arrow C stream.
    nb::object arrow_c_stream(nb::object self, bool allow_fast = true);

   private:
    // Ensures a row is available at the cursor, advancing to the next chunk as
    // needed. Returns false once the result is fully consumed.
    bool ensure_row();
    // Builds the tuple at the cursor and advances the cursor by one row.
    nb::object build_row();
    void release_chunk();
    // Backs fetch_chunk()/fetch_numpy_chunk(): returns the next data chunk,
    // or nullptr when exhausted, and accounts its rows against rows_returned_.
    duckdb_data_chunk fetch_raw_chunk();
    // Reuses a buffer from chunk_pool_ via duckdb_fetch_chunk_into when one is
    // available (no allocation); otherwise a fresh duckdb_fetch_chunk.
    duckdb_data_chunk pull_raw_chunk();

    duckdb_result result_;
    std::shared_ptr<DuckDBHandle> handle_;
    idx_t column_count_;
    // Shared with every Chunk this Result spawns; never mutated after the
    // Result constructor finishes.
    std::shared_ptr<const ChunkSchema> schema_;
    std::vector<duckdb_type> types_;
    // Per-column logical types, owned for the Result's lifetime so the row
    // decoder doesn't allocate one per cell (a column's type is constant).
    std::vector<duckdb_logical_type> column_types_;

    // Per-column state for the current chunk.
    struct ColumnCursor {
        duckdb_vector vector;
        void* data;
        uint64_t* validity;
    };

    duckdb_data_chunk chunk_ = nullptr;
    idx_t chunk_size_ = 0;
    idx_t cursor_ = 0;
    // Buffer pool for chunk-returning APIs.
    std::shared_ptr<ChunkPool> chunk_pool_ = std::make_shared<ChunkPool>();
    // Rows consumed through any cursor.
    idx_t rows_returned_ = 0;
    std::vector<ColumnCursor> cursors_;
};
