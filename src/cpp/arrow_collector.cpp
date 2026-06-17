#include "arrow_collector.hpp"

#include <cstdlib>
#include <string>
#include <vector>

// The one place ducky reaches into DuckDB's C++ internals. Everything below the
// C API is here so the blast radius of an internal-API change on a bump is this
// single translation unit. See the pivot note at the bottom of the file.
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_query_result.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/arrow/physical_arrow_collector.hpp"
#include "duckdb/main/capi/capi_internal.hpp"
#include "duckdb/main/client_config.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/main/connection.hpp"
#include "ducky.hpp"

namespace {

// Stream backing: a vector of standalone Arrow arrays the collector already
// built. Unlike result.cpp's stream, these arrays own their buffers outright
// (born as Arrow, not views into a duckdb_result), so the stream keeps nothing
// else alive — no owner reference, no connection dependency.
struct CollectedStreamState {
    std::vector<duckdb::unique_ptr<duckdb::ArrowArrayWrapper>> arrays;
    size_t index = 0;
    // Kept to regenerate the schema on demand (get_schema may be called before
    // any batch is pulled, and the arrays carry no schema of their own).
    duckdb::vector<duckdb::LogicalType> types;
    duckdb::vector<duckdb::string> names;
    duckdb::ClientProperties client_properties;
    std::string last_error;
};

int stream_get_schema(ArrowArrayStream* stream, ArrowSchema* out) {
    auto* state = (CollectedStreamState*)stream->private_data;
    try {
        duckdb::ArrowConverter::ToArrowSchema(out, state->types, state->names,
                                              state->client_properties);
    } catch (const std::exception& e) {
        state->last_error = e.what();
        return EIO;
    }
    return 0;
}

int stream_get_next(ArrowArrayStream* stream, ArrowArray* out) {
    auto* state = (CollectedStreamState*)stream->private_data;
    out->release = nullptr;  // Arrow signals end-of-stream with a released array.
    if (state->index >= state->arrays.size()) return 0;
    // Transfer ownership of the next array to the importer: move the struct and
    // null the source's release so the wrapper's destructor won't double-free.
    ArrowArray& src = state->arrays[state->index++]->arrow_array;
    *out = src;
    src.release = nullptr;
    return 0;
}

const char* stream_get_last_error(ArrowArrayStream* stream) {
    auto* state = (CollectedStreamState*)stream->private_data;
    return state->last_error.empty() ? nullptr : state->last_error.c_str();
}

void stream_release(ArrowArrayStream* stream) {
    auto* state = (CollectedStreamState*)stream->private_data;
    delete state;  // releases any unconsumed arrays via their wrappers' dtors
    stream->private_data = nullptr;
    stream->release = nullptr;
}

void stream_capsule_destructor(PyObject* capsule) {
    auto* stream = (ArrowArrayStream*)PyCapsule_GetPointer(capsule, "arrow_array_stream");
    if (stream) {
        if (stream->release) stream->release(stream);
        std::free(stream);
    }
}

}  // namespace

CollectorGuard::CollectorGuard(duckdb_connection con, uint64_t rows_per_batch) : con_(con) {
    // duckdb_connection is a plain reinterpret_cast of a C++ Connection* (see
    // duckdb-c.cpp duckdb_connect); Connection exposes its ClientContext.
    auto* connection = (duckdb::Connection*)con;
    auto& config = duckdb::ClientConfig::GetConfig(*connection->context);
    // Stash the prior hook on the heap (type-erased so the header stays clean),
    // then install the Arrow collector. Restored + freed in the destructor.
    previous_ = new duckdb::get_result_collector_t(config.get_result_collector);
    idx_t batch_size = (idx_t)rows_per_batch;
    config.get_result_collector = [batch_size](duckdb::ClientContext& c,
                                               duckdb::PreparedStatementData& data) {
        return duckdb::PhysicalArrowCollector::Create(c, data, batch_size);
    };
}

CollectorGuard::~CollectorGuard() {
    auto* connection = (duckdb::Connection*)con_;
    auto* previous = (duckdb::get_result_collector_t*)previous_;
    duckdb::ClientConfig::GetConfig(*connection->context).get_result_collector = *previous;
    delete previous;
}

nb::object arrow_stream_from_result(duckdb_result* result) {
    auto* result_data = (duckdb::DuckDBResultData*)result->internal_data;
    duckdb::QueryResult* query_result = result_data ? result_data->result.get() : nullptr;
    if (!query_result || query_result->type != duckdb::QueryResultType::ARROW_RESULT) {
        duckdb_destroy_result(result);
        throw DuckyError(
            "ducky: expected an Arrow result from the collector (statement "
            "produced no result set?)");
    }
    auto& arrow_result = (duckdb::ArrowQueryResult&)*query_result;

    auto* state = new CollectedStreamState{};
    state->arrays = arrow_result.ConsumeArrays();
    state->types = arrow_result.types;
    state->names = arrow_result.names;
    state->client_properties = arrow_result.client_properties;
    // Arrays are standalone now; the duckdb_result has nothing left we need.
    duckdb_destroy_result(result);

    auto* stream = (ArrowArrayStream*)std::calloc(1, sizeof(ArrowArrayStream));
    if (!stream) {
        delete state;
        throw std::bad_alloc();
    }
    stream->get_schema = stream_get_schema;
    stream->get_next = stream_get_next;
    stream->get_last_error = stream_get_last_error;
    stream->release = stream_release;
    stream->private_data = state;

    PyObject* capsule = PyCapsule_New(stream, "arrow_array_stream", stream_capsule_destructor);
    if (!capsule) {
        stream_release(stream);
        std::free(stream);
        throw nb::python_error();
    }
    return nb::steal(capsule);
}

// ---------------------------------------------------------------------------
// Pivoting to Route B (C-API patch, upstream-submittable)
// ---------------------------------------------------------------------------
// This file calls DuckDB's C++ collector directly because ducky links
// duckdb_static and compiles against its headers, so there's no ABI boundary —
// only source coupling to internals, contained to this TU.
//
// To turn this into the C-API patch from duckdb-arrow-collector-request.md
// (e.g. to submit upstream, or to drop the internal-header coupling entirely),
// move the two halves into ext/duckdb and carry them in patches/:
//
//   1. CollectorGuard becomes `duckdb_pending_prepared_arrow(stmt,
//      rows_per_batch, &pending)`, appended to src/main/capi/pending-c.cpp
//      (mirrors duckdb_pending_prepared_internal; set the hook via
//      wrapper->statement->context, restore right after PendingQuery).
//   2. arrow_stream_from_result becomes `duckdb_result_arrow_array_stream(
//      &result, out_stream)`, appended to src/main/capi/arrow-c.cpp (downcast
//      DuckDBResultData::result to ArrowQueryResult, ConsumeArrays, same stream
//      callbacks as here).
//   3. Declare both in src/include/duckdb/main/capi/duckdb.h.
//
// Then ducky calls those C functions and drives the pending loop against them,
// and this file (plus its internal #includes) is deleted. The stream-callback
// code ports over near-verbatim; the producer logic is identical either way.
