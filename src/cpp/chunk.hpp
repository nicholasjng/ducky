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
// a zero-copy nb::ndarray (numpy framework, DLPack-compatible).
//
// Supported types:
//   Primitive: BOOLEAN, [U]TINYINT/SMALLINT/INTEGER/BIGINT, FLOAT, DOUBLE
//   Temporal:  DATE (int32 days), TIME/TIMESTAMP*/TIMESTAMP_TZ (int64 micros)
//   Structured (numpy structured dtype, zero-copy via .view()):
//     HUGEINT  → dtype([('lower','u8'),('upper','i8')])
//     UHUGEINT → dtype([('lower','u8'),('upper','u8')])
//     INTERVAL → dtype([('months','i4'),('days','i4'),('micros','i8')])
//     DECIMAL  → raw integer matching internal storage (same structured dtype
//                for high-precision DECIMAL backed by HUGEINT); call
//                decimal_scale() to recover the exponent.
//
// String, list, struct, map and other non-flat types are not supported here —
// go through the Arrow path (`Result.arrow()`) for those.
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
    // view over the chunk's buffer. See class comment for supported types.
    nb::object column(nb::object key, nb::handle owner);

    // Returns the scale (number of decimal digits after the point) for a
    // DECIMAL column. Raises DuckyError if the column is not DECIMAL.
    int decimal_scale(nb::object key);

    // Returns the column at `key` as a DLPack-compatible object (an
    // nb_ndarray with __dlpack__ / __dlpack_device__), without going through
    // numpy. Only supported for flat numeric/temporal types; raises for
    // structured types (HUGEINT, INTERVAL, DECIMAL, VARCHAR, ...).
    nb::object dlpack(nb::object key, nb::handle owner);

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
