#pragma once

#include "duckdb.h"

// Shared DuckDB database and connection owner.
struct DuckDBHandle {
    duckdb_database database = nullptr;
    duckdb_connection connection = nullptr;

    DuckDBHandle() = default;
    DuckDBHandle(const DuckDBHandle&) = delete;
    DuckDBHandle& operator=(const DuckDBHandle&) = delete;

    ~DuckDBHandle() {
        if (connection) duckdb_disconnect(&connection);
        if (database) duckdb_close(&database);
    }
};
