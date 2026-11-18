#pragma once

#include <duckdb.h>

#include <memory>
#include <string>
#include <vector>

#include "arrow_abi.h"

// Always destroys the error_data, so callers cannot leak it on either path.
inline bool drain_arrow_error(duckdb_error_data err, std::string& out) {
    if (!err) return false;
    bool has = duckdb_error_data_has_error(err);
    if (has) out = duckdb_error_data_message(err);
    duckdb_destroy_error_data(&err);
    return has;
}

// Releases owned Arrow C structs on scope exit, including on exceptions.
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

// Filled through `&guard.schema` as an out-parameter.
struct ConvertedSchemaRelease {
    duckdb_arrow_converted_schema schema = nullptr;
    ~ConvertedSchemaRelease() {
        if (schema) duckdb_destroy_arrow_converted_schema(&schema);
    }
};

// Windows share the batch, so the last one releases it.
inline std::shared_ptr<ArrowArray> share_batch(ArrowArray& batch) {
    auto* owned = new ArrowArray(batch);
    batch.release = nullptr;
    return std::shared_ptr<ArrowArray>(owned, [](ArrowArray* a) {
        if (a->release) a->release(a);
        delete a;
    });
}

// A window's columns: views of the batch's, which the window keeps alive.
struct ArrowWindow {
    std::shared_ptr<ArrowArray> batch;
    std::vector<ArrowArray> columns;
    std::vector<ArrowArray*> column_ptrs;
};

inline void release_window(ArrowArray* array) {
    delete (ArrowWindow*)array->private_data;
    array->private_data = nullptr;
    array->release = nullptr;
}

// The batch owns the columns, so a view's release only marks it released.
inline void release_column_view(ArrowArray* array) { array->release = nullptr; }

// Sliced as Arrow slices a record batch, so it stays zero-copy: each column's
// offset and length change, and nested arrays follow their parent's offset.
inline ArrowArray window_of(const std::shared_ptr<ArrowArray>& batch, idx_t offset, idx_t count) {
    auto* w = new ArrowWindow{batch, {}, {}};
    w->columns.reserve((size_t)batch->n_children);
    for (int64_t c = 0; c < batch->n_children; ++c) {
        ArrowArray column = *batch->children[c];
        column.offset += (int64_t)(batch->offset + offset);
        column.length = (int64_t)count;
        if (column.null_count != 0) column.null_count = -1;
        column.release = release_column_view;
        column.private_data = nullptr;
        w->columns.push_back(column);
    }
    for (ArrowArray& column : w->columns) w->column_ptrs.push_back(&column);
    ArrowArray view = *batch;
    view.offset = 0;
    view.length = (int64_t)count;
    view.null_count = 0;
    view.children = w->column_ptrs.empty() ? nullptr : w->column_ptrs.data();
    view.private_data = w;
    view.release = release_window;
    return view;
}
