#pragma once

#include <duckdb.h>
#include <nanobind/nanobind.h>

#include <atomic>
#include <stdexcept>
#include <utility>

namespace nb = nanobind;

// Exception thrown across the binding layer. Registered with nanobind in
// ducky.cpp so it surfaces in Python as `ducky.Error`.
struct DuckyError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Lazily construct a process-wide singleton of type T from `init`, caching it
// behind an atomic pointer guarded by a free-threading mutex. The instance is
// intentionally leaked so its destructor never runs at interpreter shutdown,
// when the Python objects it captures may already be torn down.
//
// `init` runs at most once per successful publish and may release the GIL (it
// typically imports modules); we never hold `mu` across it, so a thread racing
// in can't deadlock against the importing thread. On a lost race the value we
// built is discarded. Callers own the `cached`/`mu` statics so each singleton
// is independent. Requires the GIL on entry (the discard path decrefs T).
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

// Run `body()` and funnel any C++ or Python exception into DuckDB's per-callback
// error channel via `set_error(const char*)`. Returns true when `body` ran to
// completion, false when it threw (and the error was reported). Used by the UDF
// / aggregate / table trampolines, which run with the GIL held and must never
// let an exception unwind across the C ABI boundary back into DuckDB.
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
