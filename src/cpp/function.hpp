#pragma once

#include <nanobind/nanobind.h>

#include <string>

#include "duckdb.h"

namespace nb = nanobind;

// Forward decl; full type in connection.hpp.
class Connection;

// Registers a Python callable as a DuckDB scalar function on `con`. Inputs are
// passed to `fn` as zero-copy 1-D numpy ndarrays over the input chunk's vectors;
// `fn` returns one ndarray of length chunk_size and matching dtype, which is
// memcpy'd into DuckDB's output vector.
//
// `parameters` is either:
//   * a list of type strings — `fn` is called positionally: fn(x, y, ...)
//   * a dict of {name: type_string} — `fn` is called with a single dict
//     {name: ndarray}. Names are Python-side labels only.
//
// NULL handling: input ndarrays are raw buffers (NULL slots hold garbage);
// filter with SQL `WHERE x IS NOT NULL` if you need clean inputs. Output is
// always non-null in v1.
void create_scalar_function(Connection& con, const std::string& name, nb::callable fn,
                            nb::object parameters, const std::string& return_type);

// Returns the duckdb_type enum for one of our supported primitive type names.
// Throws DuckyError on unknown names.
duckdb_type parse_type_name(const std::string& name);
