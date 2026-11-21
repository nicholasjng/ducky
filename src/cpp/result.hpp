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

// Resolved once per column, so nested types, field names and enum dictionaries
// are not looked up per cell.
struct CellDecoder {
    duckdb_type type = DUCKDB_TYPE_INVALID;
    duckdb_type internal = DUCKDB_TYPE_INVALID;  // DECIMAL and ENUM storage
    uint8_t scale = 0;                           // DECIMAL
    idx_t array_size = 0;                        // ARRAY
    // The element of a LIST or ARRAY, the fields of a STRUCT, key and value of a
    // MAP, the members of a UNION.
    std::vector<CellDecoder> children;
    std::vector<nb::object> names;        // STRUCT field names
    std::vector<nb::object> enum_values;  // ENUM dictionary
    // TIME_TZ: datetime.timezone per UTC offset seen.
    tsl::robin_map<int32_t, nb::object> timezones;

    // Requires the GIL.
    explicit CellDecoder(duckdb_logical_type logical);
};

// Taken whole from the result, so the rows are held once and scanned zero-copy.
class TakenRows {
   public:
    TakenRows(duckdb_v2_column_data_collection_handle collection,
              std::vector<duckdb_logical_type>& types);
    ~TakenRows();

    TakenRows(const TakenRows&) = delete;
    TakenRows& operator=(const TakenRows&) = delete;

    idx_t remaining() const { return remaining_; }
    // Valid until the next scan, since it may reference the collection's
    // buffers.
    duckdb_data_chunk scan();

   private:
    duckdb_v2_column_data_collection_handle collection_;
    duckdb_v2_column_data_collection_shared_scan_state_handle shared_ = nullptr;
    duckdb_v2_column_data_collection_worker_scan_state_handle worker_ = nullptr;
    duckdb_data_chunk scratch_ = nullptr;
    idx_t remaining_ = 0;
};

// Runs a result to its end so execute frees the connection, unless it streams
// rows, which run as they are fetched and so settle at once.
class Execution {
   public:
    Execution(duckdb_v2_result_handle result, bool streaming);
    ~Execution();

    Execution(const Execution&) = delete;
    Execution& operator=(const Execution&) = delete;

    // Takes at least one step, so a short deadline still makes progress, and
    // waits while workers own the remaining tasks. Call without the GIL, since
    // workers may run Python callbacks.
    bool run(std::chrono::steady_clock::time_point deadline);
    bool settled() const { return settled_; }

   private:
    friend class Result;

    // Taking the rows avoids stepping a copy of each chunk out of them.
    bool take_rows();

    duckdb_v2_result_handle result_;
    bool settled_;
    bool finished_ = false;
    bool retained_;
    // v1 views, since the conversions still read v1 chunks.
    std::deque<duckdb_data_chunk> chunks_;
    duckdb_v2_column_data_collection_handle rows_ = nullptr;
};

// Borrowed from the Result when `owned` is false, which saves a copy for
// consumers that read it before taking the next one.
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

inline constexpr const char* kInterrupted = "ducky: query interrupted";

// Ending an unfinished query takes the client context lock, which a query
// running Python callbacks on another thread may hold while it waits for the
// GIL.
void destroy_v2_result(duckdb_v2_result_handle& result);

class Result {
   public:
    // `handle` keeps the database alive.
    Result(Execution& execution, std::shared_ptr<DuckDBHandle> handle);
    ~Result();

    Result(const Result&) = delete;
    Result& operator=(const Result&) = delete;

    const std::vector<std::string>& column_names() const { return schema_->names; }
    std::vector<std::string> column_types() const;
    // PEP 249 Cursor.description, None without a result set.
    nb::object description() const;

    nb::object fetchone();
    nb::list fetchmany(int64_t size);
    nb::list fetchall();
    nb::object fetchitem();

    // Read when the query ran. 0 for statements that report no count (SELECT,
    // DDL).
    int64_t rows_changed() const { return rows_changed_; }

    // Shares the cursor with the other consumers.
    nb::object fetch_chunk();

    // Zero-copy views, except that `fill_nulls` copies a column with NULLs to
    // fill them as to_numpy() does.
    nb::object fetch_numpy_chunk(bool fill_nulls);

    // NumPy output holds ENUM codes, which index these labels.
    nb::list categories(nb::object key) const;

    nb::object to_numpy(nb::object columns);
    // Drains into one dict of arrays per fold: fold k takes the rows with
    // bounds[k] <= bucket < bounds[k + 1] of the BIGINT column `bucket`, which
    // is not returned. `groups` (dict of name to column names) stacks scalar
    // columns of one type into (rows, n) matrices.
    nb::object to_numpy_folds(const std::string& bucket, std::vector<int64_t> bounds,
                              nb::object groups);

    nb::object arrow_c_stream(nb::object self, bool allow_fast = true);

    // Shared with the Arrow stream callbacks, which consumers may call from any
    // thread, with or without the GIL.
    //
    // Waits with the GIL released, so a holder waiting for the GIL (or for
    // DuckDB workers that need it) cannot deadlock with the waiter.
    std::unique_lock<std::mutex> lock_cursor();
    // The row cursor's buffered rows come first, so no consumer skips rows
    // another left. Requires the cursor lock.
    duckdb_data_chunk take_chunk_locked();
    // For consumers that copy the rows out before taking more, which saves the
    // copy take_chunk_locked() makes. Requires the cursor lock.
    BorrowedChunk borrow_chunk_locked();
    // nullopt while a streaming query runs or after it failed. Requires the
    // cursor lock.
    std::optional<idx_t> remaining_rows_locked() const;

   private:
    bool ensure_row();
    nb::object build_row();
    void release_chunk();
    // Fetches with the GIL released. Requires the cursor lock.
    duckdb_data_chunk next_query_chunk_locked();
    duckdb_data_chunk copy_rows(duckdb_data_chunk chunk, idx_t offset, idx_t count);
    // Keeps `error` to raise from every later fetch.
    void end_stream(std::string error = "");

    // nullptr once every chunk is buffered or consumed.
    duckdb_v2_result_handle stream_ = nullptr;
    std::deque<duckdb_data_chunk> buffered_;
    idx_t buffered_rows_ = 0;
    // Read after buffered_.
    std::unique_ptr<TakenRows> taken_;
    std::string error_;
    // NumPy output is a dict by column name.
    void require_distinct_names() const;
    std::shared_ptr<DuckDBHandle> handle_;
    idx_t column_count_ = 0;
    int64_t rows_changed_ = 0;
    // Shared with every Chunk this Result spawns, so it is never mutated after
    // construction.
    std::shared_ptr<const ChunkSchema> schema_;
    std::vector<duckdb_type> types_;
    // Owned for the Result's lifetime so the row decoder doesn't allocate one
    // per cell.
    std::vector<duckdb_logical_type> column_types_;
    // Arrow export creates its exporter from it.
    duckdb_v2_schema_handle schema_v2_ = nullptr;

    struct ColumnCursor {
        duckdb_vector vector;
        void* data;
        uint64_t* validity;
    };

    duckdb_data_chunk chunk_ = nullptr;
    // taken_'s chunk, which the next scan overwrites.
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
