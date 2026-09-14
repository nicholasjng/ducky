#pragma once

#include <duckdb.h>
#include <nanobind/nanobind.h>

#include <atomic>
#include <stdexcept>
#include <string>
#include <utility>

namespace nb = nanobind;

// Safely copy nullable DuckDB error strings.
inline std::string duckdb_message(const char* msg) {
    return msg ? std::string(msg) : std::string("unknown DuckDB error");
}

// Python-visible binding error.
struct DuckyError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Delete a Python-owning callback context while holding the GIL.
template <typename T>
void gil_delete(void* ptr) {
    if (!ptr) return;
    nb::gil_scoped_acquire gil;
    delete (T*)ptr;
}

// Appender sentinel for a column's DEFAULT expression.
struct DefaultSentinel {};

// Publish a lazily built, process-lifetime singleton without locking across init.
template <typename T, typename Init>
const T& cached_singleton(std::atomic<T*>& cached, nb::ft_mutex& mu, Init&& init) {
    if (T* p = cached.load()) return *p;
    T* built = new T(std::forward<Init>(init)());
    nb::ft_lock_guard lock(mu);
    if (T* p = cached.load()) {  // lost the race; discard ours
        delete built;
        return *p;
    }
    cached.store(built);
    return *built;
}

// Report callback exceptions through DuckDB instead of crossing the C ABI.
template <typename Body, typename SetError>
bool guard(Body&& body, SetError&& set_error) {
    try {
        std::forward<Body>(body)();
        return true;
    } catch (nb::python_error& e) {
        set_error(e.what());
        return false;
    } catch (const std::exception& e) {
        set_error(e.what());
        return false;
    }
}

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
