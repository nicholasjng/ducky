#pragma once

#include <nanobind/nanobind.h>
#include <tsl/robin_map.h>

#include <memory>
#include <mutex>
#include <string>

#include "duckdb.h"

namespace nb = nanobind;

class Connection;

// Per-connection reusable Arrow sources for replacement scans.
struct ArrowRegistry {
    std::mutex mu;
    tsl::robin_map<std::string, nb::object> sources;
    duckdb_connection con = nullptr;  // borrowed; for duckdb_data_chunk_from_arrow
};

// Install Arrow table and replacement scans, returning their shared registry.
std::shared_ptr<ArrowRegistry> install_arrow_scan(Connection& con);
