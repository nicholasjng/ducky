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

// Owns a `duckdb_result` and decodes it into Python objects on demand.
//
// Rows are pulled lazily, one DuckDB data chunk (a column-major batch of up to
// STANDARD_VECTOR_SIZE rows) at a time, via the modern duckdb_fetch_chunk API.
// We keep the current chunk plus a cursor into it, so fetchone/fetchmany stream
// without materializing the whole result up front.
class Result {
   public:
    // `handle` is the shared database/connection owner; the Result keeps it
    // alive so that fetching and Arrow export keep working even after the
    // originating Connection is dropped.
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

    // Pulls the next data chunk and returns it as a `Chunk` Python object, or
    // None when the result is exhausted. Each call advances the same underlying
    // cursor as fetch*/arrow_c_stream — don't mix them on one result.
    nb::object fetch_chunk();

    // Pulls the next data chunk and returns it as a {name: ndarray} dict (the
    // zero-copy views Chunk.column produces), or None when exhausted. Builds
    // the whole dict in one C++ call so streaming consumers (iter_batches) pay
    // a single Python<->C++ crossing per chunk instead of one per column.
    // Shares the cursor with fetch_chunk/fetch*; don't mix them on one result.
    nb::object fetch_numpy_chunk();

    // Drain the result into a {name: numpy.ndarray} dict. Loops chunks in C++,
    // pre-allocates one buffer per column, memcpy's each chunk's vector data
    // into the right slice, and wraps the buffer as a numpy ndarray (HUGEINT /
    // UHUGEINT / INTERVAL / DECIMAL-HUGEINT get reinterpreted via .view() into
    // a structured dtype). `columns` is an Iterable[str] or None for all.
    // Raises DuckyError on column types with no flat ndarray representation
    // (VARCHAR, LIST, STRUCT, MAP) — callers should route to .arrow() instead.
    nb::object to_numpy(nb::object columns);

    // Arrow PyCapsule interface: exports the remaining result as an Arrow C
    // stream. `self` is the owning Python object, kept alive for the stream's
    // lifetime. Consumes the result (mutually exclusive with the fetch* path).
    //
    // `allow_fast` (the default, used by __arrow_c_stream__) lets all-flat
    // results be drained into a single combined batch — cheapest for whole-table
    // consumers. Pass false for a per-chunk stream (one batch per ~2048-row
    // chunk), which keeps peak memory bounded to one batch for streaming.
    nb::object arrow_c_stream(nb::object self, bool allow_fast = true);

   private:
    // Ensures a row is available at the cursor, advancing to the next chunk as
    // needed. Returns false once the result is fully consumed.
    bool ensure_row();
    // Builds the tuple at the cursor and advances the cursor by one row.
    nb::object build_row();
    void release_chunk();
    // Backs fetch_chunk()/fetch_numpy_chunk(): returns the next data chunk,
    // or nullptr when exhausted. Reuses a buffer from chunk_pool_ via
    // duckdb_fetch_chunk_into when one is available (no allocation);
    // otherwise falls back to a fresh duckdb_fetch_chunk allocation.
    duckdb_data_chunk fetch_raw_chunk();

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

    // Per-column decode state for the chunk currently being decoded, resolved
    // once per (chunk, column) in ensure_row() so the row decoder makes zero
    // C API calls per cell on flat types (data/validity/type are constant
    // across a vector). `vector` stays as the entry point for nested types,
    // whose children the decoder still resolves per cell.
    struct ColumnCursor {
        duckdb_vector vector;
        void* data;
        uint64_t* validity;
    };

    duckdb_data_chunk chunk_ = nullptr;
    idx_t chunk_size_ = 0;
    idx_t cursor_ = 0;
    // Buffer pool for fetch_chunk()/fetch_numpy_chunk() (the Chunk-returning
    // API) — distinct from chunk_ above, which is ensure_row()'s own scratch
    // buffer for the row-cursor path (fetchone/fetchmany/fetchall). One pool
    // per Result, so a reused buffer always already matches this Result's
    // own column layout.
    std::shared_ptr<ChunkPool> chunk_pool_ = std::make_shared<ChunkPool>();
    // Rows handed out via fetchone/fetchmany/fetchall so far; lets fetchall
    // pre-size its list from duckdb_row_count on materialized results.
    idx_t rows_returned_ = 0;
    std::vector<ColumnCursor> cursors_;
};
