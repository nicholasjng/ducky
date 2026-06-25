#pragma once

#include <nanobind/nanobind.h>

#include <string>
#include <vector>

#include "duckdb.h"
#include "function.hpp"

namespace nb = nanobind;

class Connection;

struct TableUDFContext {
    nb::callable factory;
    std::vector<duckdb_logical_type> param_types;  // owned
    std::vector<std::string> col_names;
    std::vector<duckdb_logical_type> col_types;  // owned
    std::vector<duckdb_type> col_type_ids;

    TableUDFContext() = default;
    TableUDFContext(const TableUDFContext&) = delete;
    TableUDFContext& operator=(const TableUDFContext&) = delete;

    // Owns the logical types so a create_table_function that throws mid-parse
    // cleans up the ones already built, not just the extra-info destroy path.
    ~TableUDFContext() {
        for (duckdb_logical_type& t : param_types) duckdb_destroy_logical_type(&t);
        for (duckdb_logical_type& t : col_types) duckdb_destroy_logical_type(&t);
    }
};

struct TableUDFBindData {
    nb::callable factory;
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
