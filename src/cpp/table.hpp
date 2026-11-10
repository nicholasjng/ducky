#pragma once

#include <nanobind/nanobind.h>

#include <string>
#include <vector>

#include "duckdb.h"
#include "function.hpp"

namespace nb = nanobind;

class Connection;

struct TableUDFContext {
    nb::handle factory;                            // kept alive by the DuckDBHandle
    std::vector<duckdb_logical_type> param_types;  // owned
    std::vector<std::string> col_names;
    std::vector<duckdb_logical_type> col_types;  // owned
    std::vector<duckdb_type> col_type_ids;

    TableUDFContext() = default;
    TableUDFContext(const TableUDFContext&) = delete;
    TableUDFContext& operator=(const TableUDFContext&) = delete;

    // Owns logical types during parsing and registration.
    ~TableUDFContext() {
        for (duckdb_logical_type& t : param_types) duckdb_destroy_logical_type(&t);
        for (duckdb_logical_type& t : col_types) duckdb_destroy_logical_type(&t);
    }
};

struct TableUDFBindData {
    nb::handle factory;  // borrowed from TableUDFContext
    nb::list python_args;
    std::vector<duckdb_type> col_type_ids;
    std::vector<std::string> col_names;
};

struct TableUDFInitData {
    nb::object generator;
    bool exhausted = false;
};

void create_table_function(Connection& con, const std::string& name, nb::callable factory,
                           nb::object parameters, nb::object columns);
