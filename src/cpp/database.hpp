#pragma once

#include "duckdb.h"

// Shared owner of a DuckDB database handle and one connection to it.
//
// Both the Connection and every Result it produces hold a shared_ptr to this,
// so the disconnect + close only run once the last referrer is gone. That lets
// a Result still be consumed — including Arrow export, which dereferences the
// connection's client context and the open database — after the owning
// Connection has been dropped (e.g. `ducky.connect().sql(...).arrow()`, or an
// Arrow capsule outliving the connection).
struct DuckDBHandle {
    duckdb_database database = nullptr;
    duckdb_connection connection = nullptr;

    DuckDBHandle() = default;
    DuckDBHandle(const DuckDBHandle &) = delete;
    DuckDBHandle &operator=(const DuckDBHandle &) = delete;

    ~DuckDBHandle() {
        if (connection) duckdb_disconnect(&connection);
        if (database) duckdb_close(&database);
    }
};
