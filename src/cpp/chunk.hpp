#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "database.hpp"
#include "duckdb.h"

namespace nb = nanobind;

// Owns a single `duckdb_data_chunk` and exposes each numeric/temporal column as
// a zero-copy nb::ndarray (numpy framework, DLPack-compatible). String, list,
// struct, decimal and other non-flat types are not exposed here — go through
// the Arrow path (`Result.arrow()`) for those.
//
// The ndarray's lifetime is tied to the Python Chunk object via nanobind's
// owner mechanism: the chunk (and the database that allocated it) is kept
// alive for as long as any column view is held.
class Chunk {
   public:
    Chunk(duckdb_data_chunk chunk, std::vector<std::string> names,
          std::shared_ptr<DuckDBHandle> handle);
    ~Chunk();

    Chunk(const Chunk&) = delete;
    Chunk& operator=(const Chunk&) = delete;

    idx_t size() const { return size_; }
    const std::vector<std::string>& column_names() const { return names_; }
    std::vector<std::string> column_types() const;

    // Returns the column at `key` (int index or str name) as a 1-D ndarray
    // view over the chunk's buffer. Raises DuckyError if the column type has
    // no flat ndarray representation.
    nb::object column(nb::object key, nb::handle owner);

    // Returns a uint8 ndarray (1 = valid, 0 = null) of length `size`, or None
    // if the column has no validity mask (i.e. no nulls). The mask is unpacked
    // from DuckDB's bitmap on first access and cached.
    nb::object validity(nb::object key, nb::handle owner);

   private:
    idx_t resolve(nb::object key) const;

    duckdb_data_chunk chunk_;
    std::vector<std::string> names_;
    std::vector<duckdb_logical_type> types_;
    std::vector<duckdb_type> type_ids_;
    std::vector<duckdb_vector> vectors_;
    std::vector<std::unique_ptr<uint8_t[]>> unpacked_validity_;
    std::shared_ptr<DuckDBHandle> handle_;
    idx_t size_;
};
