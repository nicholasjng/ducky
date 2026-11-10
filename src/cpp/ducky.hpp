#pragma once

#include <duckdb.h>
#include <nanobind/nanobind.h>

#include <atomic>
#include <mutex>
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

// Releases the GIL (detaches the thread state) for the scope if this thread
// holds it. For blocking DuckDB calls reachable both from Python and from
// threads that don't hold the GIL (e.g. Arrow stream callbacks): DuckDB
// workers running Python callbacks need the GIL while the caller waits.
class release_gil_if_held {
   public:
    release_gil_if_held() : state_(PyGILState_Check() ? PyEval_SaveThread() : nullptr) {}
    ~release_gil_if_held() {
        if (state_) PyEval_RestoreThread(state_);
    }
    release_gil_if_held(const release_gil_if_held&) = delete;
    release_gil_if_held& operator=(const release_gil_if_held&) = delete;

   private:
    PyThreadState* state_;
};

// Locks `mu`, waiting with the GIL released, so a holder that waits for the
// GIL (or for DuckDB workers that need it) cannot deadlock with the waiter.
inline std::unique_lock<std::mutex> lock_releasing_gil(std::mutex& mu) {
    std::unique_lock<std::mutex> lock(mu, std::try_to_lock);
    if (!lock.owns_lock()) {
        release_gil_if_held release;
        lock.lock();
    }
    return lock;
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
        case DUCKDB_TYPE_UNION:
            return "UNION";
        case DUCKDB_TYPE_BIT:
            return "BIT";
        case DUCKDB_TYPE_TIME_TZ:
            return "TIME_TZ";
        case DUCKDB_TYPE_TIME_NS:
            return "TIME_NS";
        case DUCKDB_TYPE_BIGNUM:
            return "BIGNUM";
        case DUCKDB_TYPE_SQLNULL:
            return "NULL";
        case DUCKDB_TYPE_GEOMETRY:
            return "GEOMETRY";
        case DUCKDB_TYPE_VARIANT:
            return "VARIANT";
        case DUCKDB_TYPE_TIMESTAMP_TZ_NS:
            return "TIMESTAMP_TZ_NS";
        default:
            return "UNKNOWN";
    }
}
