#pragma once

#include <string>

#include "duckdb_v2_bridge.h"
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
