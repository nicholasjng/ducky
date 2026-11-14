#pragma once

#include <nanobind/nanobind.h>

#include <string>
#include <vector>

#include "v2.hpp"

namespace nb = nanobind;

class Connection;

struct TableUDFContext {
    nb::handle factory;  // kept alive by the DuckDBHandle
    std::vector<std::string> col_names;
    std::vector<duckdb_v2_logical_type_handle> col_types;  // owned
    std::vector<DUCKDB_V2_LOGICAL_TYPE_ID> col_type_ids;

    TableUDFContext() = default;
    TableUDFContext(const TableUDFContext&) = delete;
    TableUDFContext& operator=(const TableUDFContext&) = delete;

    ~TableUDFContext() {
        for (duckdb_v2_logical_type_handle& t : col_types) duckdb_v2_logical_type_destroy(&t);
    }
};

struct TableUDFBindData {
    nb::handle factory;  // borrowed from TableUDFContext
    nb::list python_args;
};

struct TableUDFInitData {
    nb::object generator;
    bool exhausted = false;
};

void create_table_function(Connection& con, const std::string& name, nb::callable factory,
                           nb::object parameters, nb::object columns);
