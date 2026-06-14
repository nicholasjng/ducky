#pragma once

#include <duckdb.h>

#include <string>

#include "arrow_abi.h"

// Shared helpers for the Arrow C Data/Stream interface paths (appender,
// arrow_scan, UDF trampoline in function.cpp). Header-only so the single
// definition is shared across translation units.

// Drain a duckdb_error_data into `out`; returns true if it carried an error
// (and copied its message out). The error_data is always destroyed.
inline bool drain_arrow_error(duckdb_error_data err, std::string& out) {
    if (!err) return false;
    bool has = duckdb_error_data_has_error(err);
    if (has) out = duckdb_error_data_message(err);
    duckdb_destroy_error_data(&err);
    return has;
}

// Release-on-scope-exit for Arrow C structs we own (e.g. on exception).
struct SchemaRelease {
    ArrowSchema* s;
    ~SchemaRelease() {
        if (s && s->release) s->release(s);
    }
};
struct ArrayRelease {
    ArrowArray* a;
    ~ArrayRelease() {
        if (a && a->release) a->release(a);
    }
};

// RAII for a duckdb_arrow_converted_schema: default-construct, then fill
// `.schema` (typically by passing &guard.schema as an out-param).
struct ConvertedSchemaRelease {
    duckdb_arrow_converted_schema schema = nullptr;
    ~ConvertedSchemaRelease() {
        if (schema) duckdb_destroy_arrow_converted_schema(&schema);
    }
};
