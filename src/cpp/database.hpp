#pragma once

#include "duckdb.h"
#include "ducky.hpp"

// Shared DuckDB database and connection owner.
struct DuckDBHandle {
    duckdb_database database = nullptr;
    duckdb_connection connection = nullptr;

    DuckDBHandle() = default;
    DuckDBHandle(const DuckDBHandle&) = delete;
    DuckDBHandle& operator=(const DuckDBHandle&) = delete;

    ~DuckDBHandle() {
        // Disconnecting waits for the connection's client context, and closing
        // the database destroys registered functions, whose destructors take
        // the GIL.
        release_gil_if_held release;
        if (connection) duckdb_disconnect(&connection);
        if (database) duckdb_close(&database);
    }
};
