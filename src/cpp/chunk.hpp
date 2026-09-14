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
};

// Reusable data-chunk buffers for one result.
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

// A data chunk exposing flat columns as zero-copy arrays.
class Chunk {
   public:
    // Return the buffer to `pool` on destruction when possible.
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

    // Return a flat column as an MLX array.
    nb::object mlx(nb::object key, nb::handle owner);

    // Returns a uint8 ndarray (1 = valid, 0 = null) of length `size`, or None
    // if the column has no validity mask (i.e. no nulls). The mask is unpacked
    // from DuckDB's bitmap on first access and cached.
    nb::object validity(nb::object key, nb::handle owner);

   private:
    idx_t resolve(nb::object key) const;
    // Lazily resolve and cache logical types.
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
