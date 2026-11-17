#pragma once

#include <nanobind/nanobind.h>
#include <tsl/robin_map.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "chunk.hpp"
#include "database.hpp"
#include "duckdb.h"
#include "v2.hpp"

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

// The rows of a retained query, taken whole from its result. Scans them
// zero-copy into a chunk it owns and reuses.
class TakenRows {
   public:
    // Takes `collection`, whose column types are `types`.
    TakenRows(duckdb_v2_column_data_collection_handle collection,
              std::vector<duckdb_logical_type>& types);
    ~TakenRows();

    TakenRows(const TakenRows&) = delete;
    TakenRows& operator=(const TakenRows&) = delete;

    idx_t remaining() const { return remaining_; }
    // The next chunk of rows, or nullptr at the end. Owned by this and valid
    // until the next scan: it may reference the collection's buffers.
    duckdb_data_chunk scan();

   private:
    duckdb_v2_column_data_collection_handle collection_;
    duckdb_v2_column_data_collection_shared_scan_state_handle shared_ = nullptr;
    duckdb_v2_column_data_collection_worker_scan_state_handle worker_ = nullptr;
    duckdb_data_chunk scratch_ = nullptr;
    idx_t remaining_ = 0;
};

// Runs a C API v2 result to its end on the calling thread, unless it streams
// rows: those run as they are fetched, so they settle at once. Owns the result
// until a Result takes it, and the rows stepped or taken so far.
class Execution {
   public:
    // Takes `result`.
    Execution(duckdb_v2_result_handle result, bool streaming);
    ~Execution();

    Execution(const Execution&) = delete;
    Execution& operator=(const Execution&) = delete;

    // Steps until settled() or until `deadline` passes, waiting with the
    // result while workers own the remaining tasks; one step at the least.
    // Returns settled(). Call without the GIL. Raises DuckyError when the query
    // fails or is interrupted.
    bool run(std::chrono::steady_clock::time_point deadline);
    // Whether the result can be handed to a Result.
    bool settled() const { return settled_; }

   private:
    friend class Result;

    // Takes the rows of a retained query that has run to its end. Returns
    // whether it did.
    bool take_rows();

    duckdb_v2_result_handle result_;
    bool settled_;
    bool finished_ = false;
    // Whether the result is retained, so its rows can be taken whole.
    bool retained_;
    // v1 views of the v2 chunks; see duckdb_v2_data_chunk_get_v1.
    std::deque<duckdb_data_chunk> chunks_;
    // The rows of a retained query, taken instead of stepped into chunks_.
    duckdb_v2_column_data_collection_handle rows_ = nullptr;
};

// A chunk that a consumer reads before taking the next one: owned by the
// consumer or, when `owned` is false, borrowed from the Result.
struct BorrowedChunk {
    duckdb_data_chunk chunk = nullptr;
    bool owned = false;
    BorrowedChunk() = default;
    BorrowedChunk(duckdb_data_chunk chunk, bool owned) : chunk(chunk), owned(owned) {}
    ~BorrowedChunk() {
        if (owned && chunk) duckdb_destroy_data_chunk(&chunk);
    }
    BorrowedChunk(const BorrowedChunk&) = delete;
    BorrowedChunk& operator=(const BorrowedChunk&) = delete;
    BorrowedChunk(BorrowedChunk&& other) noexcept
        : chunk(std::exchange(other.chunk, nullptr)), owned(other.owned) {}
};

// The message of a query that connection_interrupt cancelled.
inline constexpr const char* kInterrupted = "ducky: query interrupted";

// Destroys a v2 result. Ending an unfinished query cleans it up under the
// client context lock, which a query running Python callbacks on another
// thread may hold while it waits for the GIL.
void destroy_v2_result(duckdb_v2_result_handle& result);

// A lazily decoded query result.
class Result {
   public:
    // Takes the settled `execution`'s result and chunks. `handle` keeps the
    // database alive.
    Result(Execution& execution, std::shared_ptr<DuckDBHandle> handle);
    ~Result();

    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;

    const std::vector<std::string>& column_names() const { return schema_->names; }
    std::vector<std::string> column_types() const;
    // Owned by the Result.
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

    // Rows affected by an INSERT / UPDATE / DELETE: its count, read when the
    // query ran. 0 for statements that don't report changed rows (SELECT, DDL).
    int64_t rows_changed() const { return rows_changed_; }

    // Pull the next chunk. Shares the cursor with other consumers.
    nb::object fetch_chunk();

    // Pull the next chunk as NumPy arrays: zero-copy views, except that with
    // `fill_nulls` a column containing NULLs is copied with them filled as
    // to_numpy() does.
    nb::object fetch_numpy_chunk(bool fill_nulls);

    // An ENUM column's labels, indexed by the codes NumPy output holds.
    nb::list categories(nb::object key) const;

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
    // Returns a chunk of the next unconsumed rows, which the caller owns: first
    // the rows the row cursor buffered but has not returned, then the next
    // chunk. Returns nullptr at the end; raises DuckyError when the query
    // failed. Requires the cursor lock.
    duckdb_data_chunk take_chunk_locked();
    // The same rows as take_chunk_locked(), for a consumer that copies them
    // out before taking more: valid until the next take or borrow, and
    // borrowed from the Result where that saves copying them. Requires the
    // cursor lock.
    BorrowedChunk borrow_chunk_locked();
    // Exact number of unconsumed rows, or nullopt while a streaming query still
    // runs or after it failed. Requires the cursor lock.
    std::optional<idx_t> remaining_rows_locked() const;

   private:
    // Ensures a row is available at the cursor, advancing to the next chunk as
    // needed. Returns false once the result is fully consumed.
    bool ensure_row();
    // Builds the tuple at the cursor and advances the cursor by one row.
    nb::object build_row();
    void release_chunk();
    // The next chunk of the query, skipping empty ones, or nullptr at its end.
    // Fetches with the GIL released. Requires the cursor lock.
    duckdb_data_chunk next_query_chunk_locked();
    // A copy of `count` rows of `chunk` from `offset`, owned by the caller.
    duckdb_data_chunk copy_rows(duckdb_data_chunk chunk, idx_t offset, idx_t count);
    // Ends the query, keeping `error` to raise from every later fetch.
    void end_stream(std::string error = "");

    // The streaming query's result while it runs; nullptr once every chunk is
    // buffered or consumed.
    duckdb_v2_result_handle stream_ = nullptr;
    // Chunks stepped but not yet consumed, as v1 views, and their rows.
    std::deque<duckdb_data_chunk> buffered_;
    idx_t buffered_rows_ = 0;
    // The rows of a retained query, read after buffered_; nullptr once read.
    std::unique_ptr<TakenRows> taken_;
    // Set when the stream failed; raised by every later fetch.
    std::string error_;
    std::shared_ptr<DuckDBHandle> handle_;
    idx_t column_count_ = 0;
    int64_t rows_changed_ = 0;
    // Shared with every Chunk this Result spawns; never mutated after the
    // Result constructor finishes.
    std::shared_ptr<const ChunkSchema> schema_;
    std::vector<duckdb_type> types_;
    // Per-column logical types, owned for the Result's lifetime so the row
    // decoder doesn't allocate one per cell (a column's type is constant).
    std::vector<duckdb_logical_type> column_types_;
    // The result's v2 schema, which Arrow export creates its exporter from.
    duckdb_v2_schema_handle schema_v2_ = nullptr;

    // Per-column state for the current chunk.
    struct ColumnCursor {
        duckdb_vector vector;
        void* data;
        uint64_t* validity;
    };

    duckdb_data_chunk chunk_ = nullptr;
    // Whether chunk_ is taken_'s, which the next scan overwrites.
    bool chunk_borrowed_ = false;
    idx_t chunk_size_ = 0;
    idx_t cursor_ = 0;
    std::vector<ColumnCursor> cursors_;
    // Built on the first row decoded. Requires the cursor lock.
    std::vector<CellDecoder> decoders_;
    void ensure_decoders();
    // Guards the cursor state above and all fetches from stream_. Blocking
    // fetches run with the GIL released, so binding-level locks (nb::lock_self)
    // alone don't serialize consumers.
    std::mutex mu_;
};
