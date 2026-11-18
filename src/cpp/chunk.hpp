#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <tsl/robin_map.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "database.hpp"
#include "duckdb.h"

namespace nb = nanobind;

// Schema shared by every chunk of a result.
struct ChunkSchema {
    std::vector<std::string> names;
    tsl::robin_map<std::string, idx_t> name_to_idx;

    // Negative indices count from the end.
    idx_t resolve(nb::object key) const;
};

class Chunk {
   public:
    Chunk(duckdb_data_chunk chunk, std::shared_ptr<const ChunkSchema> schema,
          std::shared_ptr<DuckDBHandle> handle);
    ~Chunk();

    Chunk(const Chunk&) = delete;
    Chunk& operator=(const Chunk&) = delete;

    idx_t size() const { return size_; }
    const std::vector<std::string>& column_names() const { return schema_->names; }
    // Non-const: fills the lazy per-column type cache.
    std::vector<std::string> column_types();

    // (rows, *dims) for a fixed-size ARRAY, so tensors need no reshape.
    nb::object column(nb::object key, nb::handle owner);

    // Lets callers that hold indices (Result::fetch_numpy_chunk) skip key
    // resolution. `owner` pins the chunk buffer behind the view.
    nb::object column_at(idx_t i, nb::handle owner);

    int decimal_scale(nb::object key);

    // Skips NumPy, so frameworks import the buffer directly. Structured types
    // (HUGEINT, DECIMAL, VARCHAR, ...) have no DLPack layout and raise.
    nb::object dlpack(nb::object key, nb::handle owner);

    nb::object mlx(nb::object key, nb::handle owner);

    // The views hold ENUM codes, which index these labels.
    nb::list categories(nb::object key);

    // None without NULLs. Unpacked from DuckDB's bitmap on first access and
    // cached, since most callers never ask.
    nb::object validity(nb::object key, nb::handle owner);

   private:
    idx_t resolve(nb::object key) const { return schema_->resolve(key); }
    // False unless the values are one contiguous buffer (a flat column, or an
    // ARRAY of one), which is what a zero-copy view needs.
    bool dense(idx_t i, void*& data, nb::dlpack::dtype& dtype, std::vector<size_t>& shape);
    duckdb_vector vector(idx_t i) const { return duckdb_data_chunk_get_vector(chunk_, i); }
    duckdb_logical_type logical_type(idx_t i);
    duckdb_type type_id(idx_t i);

    duckdb_data_chunk chunk_;
    // Shared with the Result and its other chunks, so it is immutable.
    std::shared_ptr<const ChunkSchema> schema_;
    // Filled on first access.
    std::vector<duckdb_logical_type> types_;
    std::vector<duckdb_type> type_ids_;
    std::vector<std::unique_ptr<uint8_t[]>> unpacked_validity_;
    std::shared_ptr<DuckDBHandle> handle_;
    idx_t size_;
};
