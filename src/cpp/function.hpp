#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <string>
#include <vector>

#include "duckdb.h"
#include "v2.hpp"

namespace nb = nanobind;

class Connection;

struct TypeSpec {
    const char* name;
    duckdb_type type;
    DUCKDB_V2_LOGICAL_TYPE_ID v2_type;
    size_t size;
    nb::dlpack::dtype (*dtype)();
};

const TypeSpec& lookup_typespec(const std::string& name);

const TypeSpec* typespec_for(duckdb_type type);

// A fixed-size ARRAY of a flat type, nested to any depth (FLOAT[4][3]). Its
// innermost child vector holds every row's values back to back, so the column
// is an ndarray of shape (rows, *dims).
struct ArrayLayout {
    const TypeSpec* leaf = nullptr;
    std::vector<size_t> dims;  // outermost first
    size_t values_per_row = 1;
};

bool array_layout(duckdb_logical_type logical, ArrayLayout& out);

inline duckdb_vector array_leaf(duckdb_vector v, const ArrayLayout& layout) {
    for (size_t i = 0; i < layout.dims.size(); ++i) v = duckdb_array_vector_get_child(v);
    return v;
}

// `name` is for the error other types raise.
nb::list enum_categories(duckdb_logical_type logical, const std::string& name);

// Cached NumPy dtypes for structured DuckDB storage.
struct StructDtypes {
    nb::object hugeint;   // dtype([('lower','<u8'),('upper','<i8')])
    nb::object uhugeint;  // dtype([('lower','<u8'),('upper','<u8')])
    nb::object interval;  // dtype([('months','<i4'),('days','<i4'),('micros','<i8')])
};
const StructDtypes& struct_dtypes();

void create_scalar_function(Connection& con, const std::string& name, nb::callable fn,
                            nb::object parameters, nb::object return_type, nb::object varargs,
                            nb::object init, bool is_volatile, bool special_handling);

void create_arrow_scalar_function(Connection& con, const std::string& name, nb::callable fn,
                                  nb::object parameters, const std::string& return_type,
                                  bool record_batch);
