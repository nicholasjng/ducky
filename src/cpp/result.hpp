#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

    const std::vector<std::string>& column_names() const { return names_; }
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

    // Pulls the next data chunk and returns it as a `Chunk` Python object, or
    // None when the result is exhausted. Each call advances the same underlying
    // cursor as fetch*/arrow_c_stream — don't mix them on one result.
    nb::object fetch_chunk();

    // Arrow PyCapsule interface: exports the remaining result as an Arrow C
    // stream. `self` is the owning Python object, kept alive for the stream's
    // lifetime. Consumes the result (mutually exclusive with the fetch* path).
    nb::object arrow_c_stream(nb::object self);

   private:
    // Ensures a row is available at the cursor, advancing to the next chunk as
    // needed. Returns false once the result is fully consumed.
    bool ensure_row();
    // Builds the tuple at the cursor and advances the cursor by one row.
    nb::object build_row();
    void release_chunk();

    duckdb_result result_;
    std::shared_ptr<DuckDBHandle> handle_;
    idx_t column_count_;
    std::vector<std::string> names_;
    std::vector<duckdb_type> types_;
    // Per-column logical types, owned for the Result's lifetime so the row
    // decoder doesn't allocate one per cell (a column's type is constant).
    std::vector<duckdb_logical_type> column_types_;

    // State for the chunk currently being decoded: the per-column vectors are
    // the entry point for the recursive decoder (it pulls data, validity, type
    // and nested children from each vector).
    duckdb_data_chunk chunk_ = nullptr;
    idx_t chunk_size_ = 0;
    idx_t cursor_ = 0;
    std::vector<duckdb_vector> vectors_;
};
