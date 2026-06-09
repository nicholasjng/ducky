#pragma once

#include <nanobind/nanobind.h>
#include <tsl/robin_map.h>

#include <memory>
#include <mutex>
#include <string>

#include "duckdb.h"

namespace nb = nanobind;

class Connection;

// Per-connection registry backing the lazy Arrow replacement scan. Maps a table
// name to a Python object exposing __arrow_c_stream__; the source is re-streamed
// on each query rather than materialized, so repeated queries see live data (the
// source must therefore support being streamed more than once). Created on the
// first register_arrow and owned by the Connection.
//
// `mu` guards the map's structure (the replacement-scan callback may run on a
// thread without the GIL and only needs to test membership). Operations that
// touch the nb::object values (insert / erase / copy-out) additionally hold the
// GIL, which the register and bind paths already do.
struct ArrowRegistry {
    std::mutex mu;
    tsl::robin_map<std::string, nb::object> sources;
    duckdb_connection con = nullptr;  // borrowed; for duckdb_data_chunk_from_arrow
};

// Register the `ducky_arrow_scan` table function and add the replacement scan on
// `con`'s database, returning the registry (owned by the Connection). Idempotent
// per connection — call once and keep the result.
std::unique_ptr<ArrowRegistry> install_arrow_scan(Connection& con);
