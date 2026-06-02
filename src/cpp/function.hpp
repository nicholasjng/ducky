#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <string>

#include "duckdb.h"

namespace nb = nanobind;

// Forward decl; full type in connection.hpp.
class Connection;

// A primitive type supported by ndarray UDFs and aggregate UDFs: DuckDB type
// name, enum value, element size in bytes, and DLPack dtype factory.
struct TypeSpec {
    const char* name;
    duckdb_type type;
    size_t size;
    nb::dlpack::dtype (*dtype)();
};

// Returns the TypeSpec for `name`, or throws DuckyError if not found.
// Covers the same set as Chunk.column(): BOOLEAN, [U]TINYINT..BIGINT,
// FLOAT, DOUBLE, DATE, TIME, TIMESTAMP variants.
const TypeSpec& lookup_typespec(const std::string& name);

// Returns the TypeSpec for a DuckDB primitive type id, or nullptr if the type
// has no flat ndarray representation (VARCHAR, nested, DECIMAL, HUGEINT, ...).
// Single source of truth for the type → dtype/size mapping shared by the
// chunk exporter (chunk.cpp) and the columnar appender (appender.cpp).
const TypeSpec* typespec_for(duckdb_type type);

// Parse a DuckDB type expression via a SQL round-trip. Supports the full type
// grammar: "VARCHAR", "DECIMAL(10,2)", "LIST(INTEGER)", etc.
// Returns an owned logical type; caller must destroy with duckdb_destroy_logical_type.
duckdb_logical_type parse_logical_type(duckdb_connection con, const std::string& type_str);

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
