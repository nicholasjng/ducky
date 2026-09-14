#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <string>

#include "duckdb.h"

namespace nb = nanobind;

// Forward decl; full type in connection.hpp.
class Connection;

// Flat DuckDB and DLPack type metadata.
struct TypeSpec {
    const char* name;
    duckdb_type type;
    size_t size;
    nb::dlpack::dtype (*dtype)();
};

// Return a supported type or throw.
const TypeSpec& lookup_typespec(const std::string& name);

// Return flat type metadata, or nullptr when unsupported.
const TypeSpec* typespec_for(duckdb_type type);

// Cached NumPy dtypes for structured DuckDB storage.
struct StructDtypes {
    nb::object hugeint;   // dtype([('lower','<u8'),('upper','<i8')])
    nb::object uhugeint;  // dtype([('lower','<u8'),('upper','<u8')])
    nb::object interval;  // dtype([('months','<i4'),('days','<i4'),('micros','<i8')])
};
const StructDtypes& struct_dtypes();

// Parse a type expression; the caller owns the result.
duckdb_logical_type parse_logical_type(duckdb_connection con, const std::string& type_str);

// Register a chunk-wise NumPy scalar UDF.
void create_scalar_function(Connection& con, const std::string& name, nb::callable fn,
                            nb::object parameters, nb::object return_type, nb::object varargs,
                            nb::object init, bool is_volatile, bool special_handling);

// Register an Arrow scalar UDF.
void create_arrow_scalar_function(Connection& con, const std::string& name, nb::callable fn,
                                  nb::object parameters, const std::string& return_type,
                                  bool record_batch);
