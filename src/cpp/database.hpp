#pragma once

#include <atomic>
#include <cstdint>
#include <vector>

#include "duckdb.h"
#include "ducky.hpp"
#include "v2.hpp"

// Shared DuckDB database and connection owner. The instance and connection are
// C API v2 handles; `database` and `connection` bridge them to v1 for the
// functions v2 lacks.
struct DuckDBHandle {
    duckdb_v2_instance_handle instance = nullptr;
    duckdb_v2_connection_handle connection_v2 = nullptr;
    duckdb_database database = nullptr;      // owned
    duckdb_connection connection = nullptr;  // borrowed from connection_v2
    // Bumped by every Arrow source registration. A plan reads a source's schema
    // when it is bound, so prepared statements made before a registration
    // prepare again (see PreparedStatement).
    std::atomic<uint64_t> arrow_generation{0};

    DuckDBHandle() = default;
    DuckDBHandle(const DuckDBHandle&) = delete;
    DuckDBHandle& operator=(const DuckDBHandle&) = delete;

    ~DuckDBHandle() {
        {
            // Disconnecting waits for the connection's client context, and
            // closing the database destroys registered functions, whose
            // destructors take the GIL.
            release_gil_if_held release;
            connection = nullptr;
            duckdb_v2_connection_destroy(&connection_v2);
            if (database) duckdb_close(&database);
            duckdb_v2_instance_destroy(&instance);
        }
        // The functions that borrowed these are gone now.
        if (!py_refs_.empty() && Py_IsInitialized()) {
            nb::gil_scoped_acquire gil;
            py_refs_.clear();
        }
    }

    // Keeps `obj` alive as long as the database, returning a borrowed handle.
    // For the Python callables of registered functions: the Connection reports
    // them to the garbage collector (see gc_traverse), so a callable that
    // refers back to its Connection doesn't keep both alive forever.
    nb::handle keep(nb::object obj) {
        nb::ft_lock_guard lock(py_refs_mu_);
        py_refs_.push_back(std::move(obj));
        return py_refs_.back();
    }

    // For gc_traverse only: GC holds the GIL or stops all threads.
    const std::vector<nb::object>& py_refs() const { return py_refs_; }

   private:
    nb::ft_mutex py_refs_mu_;
    std::vector<nb::object> py_refs_;
};
