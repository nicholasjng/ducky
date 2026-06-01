#pragma once

#include <duckdb.h>

#include <stdexcept>

// Exception thrown across the binding layer. Registered with nanobind in
// ducky.cpp so it surfaces in Python as `ducky.Error`.
struct DuckyError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline const char* duckdb_type_name(duckdb_type type) {
    switch (type) {
        case DUCKDB_TYPE_BOOLEAN:
            return "BOOLEAN";
        case DUCKDB_TYPE_TINYINT:
            return "TINYINT";
        case DUCKDB_TYPE_SMALLINT:
            return "SMALLINT";
        case DUCKDB_TYPE_INTEGER:
            return "INTEGER";
        case DUCKDB_TYPE_BIGINT:
            return "BIGINT";
        case DUCKDB_TYPE_UTINYINT:
            return "UTINYINT";
        case DUCKDB_TYPE_USMALLINT:
            return "USMALLINT";
        case DUCKDB_TYPE_UINTEGER:
            return "UINTEGER";
        case DUCKDB_TYPE_UBIGINT:
            return "UBIGINT";
        case DUCKDB_TYPE_HUGEINT:
            return "HUGEINT";
        case DUCKDB_TYPE_UHUGEINT:
            return "UHUGEINT";
        case DUCKDB_TYPE_FLOAT:
            return "FLOAT";
        case DUCKDB_TYPE_DOUBLE:
            return "DOUBLE";
        case DUCKDB_TYPE_DECIMAL:
            return "DECIMAL";
        case DUCKDB_TYPE_VARCHAR:
            return "VARCHAR";
        case DUCKDB_TYPE_BLOB:
            return "BLOB";
        case DUCKDB_TYPE_DATE:
            return "DATE";
        case DUCKDB_TYPE_TIME:
            return "TIME";
        case DUCKDB_TYPE_TIMESTAMP:
            return "TIMESTAMP";
        case DUCKDB_TYPE_TIMESTAMP_S:
            return "TIMESTAMP_S";
        case DUCKDB_TYPE_TIMESTAMP_MS:
            return "TIMESTAMP_MS";
        case DUCKDB_TYPE_TIMESTAMP_NS:
            return "TIMESTAMP_NS";
        case DUCKDB_TYPE_TIMESTAMP_TZ:
            return "TIMESTAMP_TZ";
        case DUCKDB_TYPE_UUID:
            return "UUID";
        case DUCKDB_TYPE_INTERVAL:
            return "INTERVAL";
        case DUCKDB_TYPE_ENUM:
            return "ENUM";
        case DUCKDB_TYPE_LIST:
            return "LIST";
        case DUCKDB_TYPE_STRUCT:
            return "STRUCT";
        case DUCKDB_TYPE_MAP:
            return "MAP";
        case DUCKDB_TYPE_ARRAY:
            return "ARRAY";
        default:
            return "UNKNOWN";
    }
}
