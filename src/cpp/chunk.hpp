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

// Shared schema (column names + name → index) for all the chunks of one
// Result. Built once by Result and held by every Chunk it spawns, so
// fetch_chunk doesn't deep-copy the name vector per chunk.
struct ChunkSchema {
    std::vector<std::string> names;
    tsl::robin_map<std::string, idx_t> name_to_idx;
};

// Per-Result pool of freed `duckdb_data_chunk` buffers. duckdb_fetch_chunk
// (the API Result::fetch_chunk/fetch_numpy_chunk use) allocates a brand-new
// DataChunk on every call; when a Chunk is destroyed it hands its buffer back
// here instead of destroying it, so the next fetch can refill it in place via
// duckdb_fetch_chunk_into (no allocation) rather than paying that cost again.
// Falls back to a fresh allocation whenever the pool is empty (e.g. a caller
// holding more than one live batch at once) — always correct, just not
// always fast. No dependency on Result, to avoid a circular header include
// (result.hpp already includes chunk.hpp).
struct ChunkPool {
    nb::ft_mutex mu_;
    std::vector<duckdb_data_chunk> free_;
    static constexpr size_t kCapacity = 1;

    ~ChunkPool() {
        for (duckdb_data_chunk c : free_) duckdb_destroy_data_chunk(&c);
    }

    // Takes ownership of `chunk` if there's room; returns false (caller keeps
    // ownership) if the pool is full.
    bool try_return(duckdb_data_chunk chunk) {
        nb::ft_lock_guard lock(mu_);
        if (free_.size() >= kCapacity) return false;
        free_.push_back(chunk);
        return true;
    }

    // Returns a previously-freed buffer, or nullptr if none is available.
    duckdb_data_chunk take() {
        nb::ft_lock_guard lock(mu_);
        if (free_.empty()) return nullptr;
        duckdb_data_chunk c = free_.back();
        free_.pop_back();
        return c;
    }
};

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
    // `pool`, if alive, receives `chunk`'s buffer back on destruction instead
    // of it being destroyed outright — see ChunkPool. Defaults to empty for
    // any construction site that doesn't have (or want) a pool.
    Chunk(duckdb_data_chunk chunk, std::shared_ptr<const ChunkSchema> schema,
          std::shared_ptr<DuckDBHandle> handle, std::weak_ptr<ChunkPool> pool = {});
    ~Chunk();

    Chunk(const Chunk&) = delete;
    Chunk& operator=(const Chunk&) = delete;

    idx_t size() const { return size_; }
    const std::vector<std::string>& column_names() const { return schema_->names; }
    // Non-const: fills the lazy per-column type cache.
    std::vector<std::string> column_types();

    // Returns the column at `key` (int index or str name) as a 1-D ndarray
    // view over the chunk's buffer. See class comment for supported types.
    nb::object column(nb::object key, nb::handle owner);

    // Same as column(), but takes a pre-resolved column index. Lets a caller
    // that already holds indices (e.g. Result::fetch_numpy_chunk) skip the
    // per-call key resolution. `owner` pins the chunk buffer behind the view.
    nb::object column_at(idx_t i, nb::handle owner);

    // Returns the scale (number of decimal digits after the point) for a
    // DECIMAL column. Raises DuckyError if the column is not DECIMAL.
    int decimal_scale(nb::object key);

    // Returns the column at `key` as a DLPack-compatible object (an
    // nb_ndarray with __dlpack__ / __dlpack_device__), without going through
    // numpy. Only supported for flat numeric/temporal types; raises for
    // structured types (HUGEINT, INTERVAL, DECIMAL, VARCHAR, ...).
    nb::object dlpack(nb::object key, nb::handle owner);

    // Returns the column at `key` as an mlx.core.array, via nanobind's native
    // mlx ndarray framework (nb::mlx, added in nanobind 2.13). Same
    // zero-copy-on-CPU path as dlpack() + mx.array() in Python, done in one
    // C++ call instead of two. Same type restrictions as dlpack().
    nb::object mlx(nb::object key, nb::handle owner);

    // Returns a uint8 ndarray (1 = valid, 0 = null) of length `size`, or None
    // if the column has no validity mask (i.e. no nulls). The mask is unpacked
    // from DuckDB's bitmap on first access and cached.
    nb::object validity(nb::object key, nb::handle owner);

   private:
    idx_t resolve(nb::object key) const;
    // Lazy per-column accessors. Chunk construction sits on the per-chunk hot
    // path (fetch_chunk / iter_batches at ~489 chunks per 1M rows), so the
    // constructor materializes no per-column state; logical types — which
    // duckdb allocates per duckdb_vector_get_column_type call — are fetched
    // and cached only for the columns actually touched.
    duckdb_vector vector(idx_t i) const { return duckdb_data_chunk_get_vector(chunk_, i); }
    duckdb_logical_type logical_type(idx_t i);
    duckdb_type type_id(idx_t i);

    duckdb_data_chunk chunk_;
    // Shared with the Result that produced this chunk; immutable across all
    // chunks of that result.
    std::shared_ptr<const ChunkSchema> schema_;
    // Lazily sized + filled by logical_type()/type_id(); empty until the
    // first per-column access.
    std::vector<duckdb_logical_type> types_;
    std::vector<duckdb_type> type_ids_;
    std::vector<std::unique_ptr<uint8_t[]>> unpacked_validity_;
    std::shared_ptr<DuckDBHandle> handle_;
    std::weak_ptr<ChunkPool> pool_;
    idx_t size_;
};
