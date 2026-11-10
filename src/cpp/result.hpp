#pragma once

#include <nanobind/nanobind.h>
#include <tsl/robin_map.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "chunk.hpp"
#include "database.hpp"
#include "duckdb.h"

namespace nb = nanobind;

// How to decode one column's cells into Python objects, resolved once per
// column: nested types, field names, enum dictionaries.
struct CellDecoder {
    duckdb_type type = DUCKDB_TYPE_INVALID;
    duckdb_type internal = DUCKDB_TYPE_INVALID;  // DECIMAL and ENUM storage
    uint8_t scale = 0;                           // DECIMAL
    idx_t array_size = 0;                        // ARRAY
    // LIST and ARRAY: the element; STRUCT: the fields; MAP: key and value;
    // UNION: the members.
    std::vector<CellDecoder> children;
    std::vector<nb::object> names;        // STRUCT field names
    std::vector<nb::object> enum_values;  // ENUM dictionary
    // TIME_TZ: datetime.timezone per UTC offset seen.
    tsl::robin_map<int32_t, nb::object> timezones;

    // Requires the GIL.
    explicit CellDecoder(duckdb_logical_type logical);
};

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

    // Cursor primitives shared with the Arrow stream callbacks, which DuckDB
    // consumers may call from any thread, with or without the GIL.
    //
    // Locks the cursor. Waits with the GIL released, so a holder that is
    // waiting for the GIL (or for DuckDB workers that need it) cannot deadlock
    // with the waiter.
    std::unique_lock<std::mutex> lock_cursor();
    // Fills `dst` (a chunk of the result's column types, see
    // make_result_chunk) with the next unconsumed rows: first the rows the row
    // cursor buffered but has not returned, then a fresh fetch. Counts them as
    // returned. Returns false at the end; raises DuckyError when the query
    // failed or the result was invalidated. Requires the cursor lock.
    bool next_chunk_locked(duckdb_data_chunk dst);
    // Exact number of unconsumed rows, or nullopt when unknown (streaming).
    // Requires the cursor lock.
    std::optional<idx_t> remaining_rows_locked();
    duckdb_result* raw_result() { return &result_; }

   private:
    // Ensures a row is available at the cursor, advancing to the next chunk as
    // needed. Returns false once the result is fully consumed.
    bool ensure_row();
    // Builds the tuple at the cursor and advances the cursor by one row.
    nb::object build_row();
    void release_chunk();
    // Backs fetch_chunk()/fetch_numpy_chunk(): returns the next data chunk
    // (reusing a buffer from chunk_pool_ when one is free), or nullptr when
    // exhausted. Requires the cursor lock.
    duckdb_data_chunk fetch_raw_chunk_locked();
    // Fetches the next chunk from DuckDB into `dst` with the GIL released.
    // Returns false at the end; raises when the result reports an error.
    bool fetch_into_locked(duckdb_data_chunk dst);

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
    // Rows handed out through any consumer (rows, chunks, arrays, Arrow).
    idx_t rows_returned_ = 0;
    // Set once DuckDB reported the end of the result.
    bool exhausted_ = false;
    // Whether the result streams. Captured at construction: DuckDB reports a
    // failed streaming result as non-streaming.
    bool streaming_ = false;
    std::vector<ColumnCursor> cursors_;
    // Built on the first row decoded. Requires the cursor lock.
    std::vector<CellDecoder> decoders_;
    void ensure_decoders();
    // Guards the cursor state above and all fetches from result_. Blocking
    // fetches run with the GIL released, so binding-level locks (nb::lock_self)
    // alone don't serialize consumers.
    std::mutex mu_;
};

// A chunk with `result`'s column types, or nullptr for a zero-column result.
duckdb_data_chunk make_result_chunk(duckdb_result* result);
