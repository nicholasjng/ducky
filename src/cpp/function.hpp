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

// Cached numpy structured dtypes used to re-view HUGEINT / UHUGEINT / INTERVAL
// memory (which has no DLPack equivalent). Built lazily on first access.
struct StructDtypes {
    nb::object hugeint;   // dtype([('lower','<u8'),('upper','<i8')])
    nb::object uhugeint;  // dtype([('lower','<u8'),('upper','<u8')])
    nb::object interval;  // dtype([('months','<i4'),('days','<i4'),('micros','<i8')])
};
const StructDtypes& struct_dtypes();

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
// NULL handling: by default, input ndarrays are raw buffers (NULL slots hold
// garbage) and DuckDB auto-NULLs the output for rows where any input is NULL.
// Pass `special_handling=true` to receive each input as a (values, mask) tuple
// (mask: 1-D uint8 ndarray, 1=valid, 0=NULL) and to take responsibility for
// emitting NULLs yourself.
//
// `init` is an optional zero-arg Python callable invoked once per worker thread
// when the UDF starts executing; its return value is passed as the first
// positional argument to `fn` on every chunk (per-thread state for RNGs,
// running counters, etc.). `is_volatile=true` marks the function as
// non-deterministic so the optimizer won't constant-fold or cache it.
void create_scalar_function(Connection& con, const std::string& name, nb::callable fn,
                            nb::object parameters, nb::object return_type, nb::object varargs,
                            nb::object init, bool is_volatile, bool special_handling);

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
