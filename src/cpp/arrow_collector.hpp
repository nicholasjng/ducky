#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>

#include "duckdb.h"

namespace nb = nanobind;

// Born-as-Arrow result collection (Route A: ducky-side C++, no DuckDB patch).
//
// Executes an already-prepared statement with DuckDB's PhysicalArrowCollector
// selected as the result collector, so the engine assembles Arrow record
// batches *during* execution instead of materializing DuckDB chunks that a
// second pass then converts to Arrow. Returns an `arrow_array_stream` PyCapsule
// (consumable by pyarrow's RecordBatchReader._import_from_c_capsule) that yields
// one RecordBatch per `rows_per_batch`-row collector batch.
//
// This is the *only* file that includes DuckDB's internal C++ headers — the
// coupling to unstable internals (PhysicalArrowCollector, ArrowQueryResult,
// DuckDBResultData) is quarantined here so a `bump-duckdb.sh` that changes those
// breaks one TU at compile time, not the codebase. See the pivot-to-Route-B
// note at the bottom of arrow_collector.cpp.
//
// Usage: hold a CollectorGuard over the execution (it installs the Arrow
// collector at plan time and restores the previous hook when destroyed), run
// the query through the normal pending/execute loop so it stays Ctrl-C
// interruptible, then hand the resulting duckdb_result to
// arrow_stream_from_result.

// RAII: install DuckDB's PhysicalArrowCollector as `con`'s result collector,
// restoring the previous hook on destruction. The hook is consumed at plan
// time (inside the pending-query step), so the guard must stay alive across
// that step; keeping it alive for the whole execution is fine (DuckDB
// serializes queries per connection). Header-opaque so connection.cpp stays
// free of DuckDB's C++ headers.
class CollectorGuard {
   public:
    CollectorGuard(duckdb_connection con, uint64_t rows_per_batch);
    ~CollectorGuard();

    CollectorGuard(const CollectorGuard&) = delete;
    CollectorGuard& operator=(const CollectorGuard&) = delete;

   private:
    duckdb_connection con_;
    void* previous_;  // heap-owned copy of the prior collector hook; restored in dtor
};

// Consume an Arrow-collected `result` (produced while a CollectorGuard was held)
// and return an `arrow_array_stream` PyCapsule over its batches. Destroys
// `result`. Requires the GIL. Raises DuckyError if `result` is not an Arrow
// result (e.g. the statement produced no result set).
nb::object arrow_stream_from_result(duckdb_result* result);
