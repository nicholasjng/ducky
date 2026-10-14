#pragma once

#include <nanobind/nanobind.h>
#include <tsl/robin_map.h>

#include <memory>
#include <mutex>
#include <string>

#include "duckdb.h"

namespace nb = nanobind;

class Connection;

// A registered Arrow source.
struct ArrowSource {
    // Reusable sources (tables, data frames) are re-streamed for every scan.
    nb::object object;
    // One-shot sources (readers, query results) yield their batches once: their
    // single stream capsule is taken at registration, and `consumed` is set when
    // a scan starts reading it.
    nb::object stream;
    bool consumed = false;
};

// Per-connection Arrow sources for replacement scans.
struct ArrowRegistry {
    ArrowRegistry() = default;
    ArrowRegistry(const ArrowRegistry&) = delete;
    ArrowRegistry& operator=(const ArrowRegistry&) = delete;
    // DuckDB may drop the last reference from a thread without the GIL.
    ~ArrowRegistry();

    std::mutex mu;
    // Keyed by the lowercased name: DuckDB resolves table names
    // case-insensitively.
    tsl::robin_map<std::string, ArrowSource> sources;
    duckdb_connection con = nullptr;  // borrowed; for Arrow conversions
};

// Install Arrow table and replacement scans, returning their shared registry.
std::shared_ptr<ArrowRegistry> install_arrow_scan(Connection& con);

// Register (or replace) `obj` as `name`. Requires the GIL.
void register_arrow_source(ArrowRegistry& registry, const std::string& name, nb::object obj);
