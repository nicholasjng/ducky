#pragma once

#include <cstring>
#include <string>
#include <utility>

#include "duckdb_v2_bridge.h"
#include "duckdb_v2_connection_arrow_exporter.h"
#include "duckdb_v2_prepared_statement_metadata.h"
#include "duckdb_v2_result_retain.h"
#include "ducky.hpp"

// The process's C API v2 environment. Never destroyed: an instance can outlive
// any module-level owner, and the environment refuses to go before them.
inline duckdb_v2_environment_handle v2_environment() {
    static duckdb_v2_environment_handle env = [] {
        duckdb_v2_environment_handle out = nullptr;
        if (duckdb_v2_environment_create(&out, nullptr) != DUCKDB_V2_ERROR_NONE) {
            throw DuckyError("ducky: failed to create the DuckDB environment");
        }
        return out;
    }();
    return env;
}

// Owns the error info a v2 call reports through its `err` slot.
class V2Error {
   public:
    V2Error() = default;
    ~V2Error() { duckdb_v2_error_info_destroy(&info_); }

    V2Error(const V2Error&) = delete;
    V2Error& operator=(const V2Error&) = delete;

    duckdb_v2_error_info_handle* out() {
        duckdb_v2_error_info_destroy(&info_);
        return &info_;
    }

    std::string text() const {
        const char* text = nullptr;
        idx_t length = 0;
        if (!info_ ||
            duckdb_v2_error_info_get_text(info_, &text, &length) != DUCKDB_V2_ERROR_NONE || !text) {
            return "unknown error";
        }
        return std::string(text, length);
    }

    // Raises DuckyError with `prefix` and the reported text unless `code` is
    // DUCKDB_V2_ERROR_NONE.
    void check(DUCKDB_V2_ERROR code, const std::string& prefix = "") const {
        if (code != DUCKDB_V2_ERROR_NONE) throw DuckyError(prefix + text());
    }

   private:
    duckdb_v2_error_info_handle info_ = nullptr;
};

// Runs a callback body, reporting what it throws through the callback's `err`.
template <typename Body>
void report(duckdb_v2_error_info_handle* err, Body&& body) {
    guard(std::forward<Body>(body), [&](const char* what) {
        duckdb_v2_error_info_set_code(*err, DUCKDB_V2_ERROR_INPUT_INVALID);
        duckdb_v2_error_info_set_text(*err, what, std::strlen(what));
    });
}

// Parses a DuckDB type expression (e.g. "DECIMAL(18,3)", "STRUCT(a INT)") in
// the scope of `factory`; the caller owns the result. A connection's factory
// resolves catalog types in its open transaction, if any.
inline duckdb_v2_logical_type_handle v2_parse_type(duckdb_v2_factory_handle factory,
                                                   const std::string& text) {
    if (text.empty()) throw DuckyError("ducky: empty type string");
    duckdb_v2_logical_type_handle out = nullptr;
    V2Error err;
    DUCKDB_V2_ERROR code;
    {
        // Resolving catalog types takes the client context lock; Python
        // callbacks of a query running elsewhere may need the GIL meanwhile.
        release_gil_if_held release;
        code = duckdb_v2_factory_create_type_from_text(factory, text.data(), text.size(), &out,
                                                       err.out());
    }
    err.check(code, "ducky: invalid DuckDB type '" + text + "': ");
    return out;
}

// The name of a type's id, such as "INTEGER" or "LIST", or its alias.
inline std::string v2_type_name(duckdb_v2_logical_type_handle type) {
    const char* name = nullptr;
    idx_t length = 0;
    if (duckdb_v2_logical_type_get_name(type, &name, &length, nullptr) != DUCKDB_V2_ERROR_NONE) {
        return "unknown";
    }
    return std::string(name, length);
}
