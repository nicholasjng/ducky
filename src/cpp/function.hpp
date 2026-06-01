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
                            nb::object parameters, nb::object return_type, nb::object varargs);

// Registers a Python callable as a DuckDB scalar function, with input passed as
// a `pyarrow.RecordBatch` and output expected as a `pyarrow.Array`. Built on
// DuckDB's Arrow C-API path, so it covers the types the ndarray UDF refuses:
// VARCHAR, LIST, STRUCT, DECIMAL, MAP, etc.
//
// `parameters` is either:
//   * a list of DuckDB type strings — column names default to "arg0", "arg1", ...
//   * a dict of {name: type_string} — names are used as RecordBatch field names.
// `return_type` is a single DuckDB type string and is required.
void create_arrow_scalar_function(Connection& con, const std::string& name, nb::callable fn,
                                  nb::object parameters, const std::string& return_type,
                                  bool record_batch);

// Returns the duckdb_type enum for one of our supported primitive type names.
// Throws DuckyError on unknown names.
duckdb_type parse_type_name(const std::string& name);
