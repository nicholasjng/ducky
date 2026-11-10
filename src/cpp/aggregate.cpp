#include "aggregate.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <tsl/robin_map.h>

#include <vector>

#include "connection.hpp"
#include "ducky.hpp"
#include "function.hpp"

namespace nb = nanobind;

namespace {

// Each state slot holds a strong reference to the class instance, or nullptr
// while the state has received no rows. Instances are created lazily, on the
// state's first update, merge or finalize, so states DuckDB initializes but
// never fills (and empty merge sources) cost no Python call.
idx_t agg_state_size(duckdb_function_info /*info*/) { return sizeof(PyObject*); }

void agg_init(duckdb_function_info /*info*/, duckdb_aggregate_state state) {
    // State memory isn't zero-initialized by DuckDB, and agg_destroy
    // unconditionally Py_XDECREFs the slot.
    state->internal_ptr = nullptr;
}

// The instance in `state`, created on first use.
nb::handle instance(AggregateUDFContext* ctx, duckdb_aggregate_state state) {
    if (!state->internal_ptr) state->internal_ptr = (void*)ctx->cls().release().ptr();
    return nb::handle((PyObject*)state->internal_ptr);
}

void agg_destroy(duckdb_aggregate_state* states, idx_t count) {
    nb::gil_scoped_acquire gil;
    for (idx_t i = 0; i < count; ++i) {
        PyObject* obj = (PyObject*)states[i]->internal_ptr;
        Py_XDECREF(obj);
        states[i]->internal_ptr = nullptr;
    }
}

void agg_update(duckdb_function_info info, duckdb_data_chunk input,
                duckdb_aggregate_state* states) {
    auto* ctx = (AggregateUDFContext*)duckdb_aggregate_function_get_extra_info(info);
    idx_t n = duckdb_data_chunk_get_size(input);
    if (n == 0) return;
    idx_t n_cols = (idx_t)ctx->param_types.size();

    // SQL aggregates ignore rows with a NULL argument; DuckDB passes them in,
    // so leave them out here. `rows` stays empty when every row is valid.
    std::vector<uint64_t*> masks;
    for (idx_t c = 0; c < n_cols; ++c) {
        if (uint64_t* m = duckdb_vector_get_validity(duckdb_data_chunk_get_vector(input, c))) {
            masks.push_back(m);
        }
    }
    std::vector<idx_t> rows;
    if (!masks.empty()) {
        rows.reserve(n);
        for (idx_t r = 0; r < n; ++r) {
            bool valid = true;
            for (uint64_t* m : masks) {
                if (!duckdb_validity_row_is_valid(m, r)) {
                    valid = false;
                    break;
                }
            }
            if (valid) rows.push_back(r);
        }
        if (rows.empty()) return;
        if (rows.size() == n) rows.clear();
    }
    bool all_rows = rows.empty();

    nb::gil_scoped_acquire gil;
    guard(
        [&] {
            // Input arrays for the rows `selection` lists, gathered into
            // owned buffers (the caller may keep the arrays).
            auto gather_args = [&](const std::vector<idx_t>& selection) {
                size_t shape[1] = {selection.size()};
                nb::list args;
                for (idx_t c = 0; c < n_cols; ++c) {
                    duckdb_vector v = duckdb_data_chunk_get_vector(input, c);
                    const uint8_t* src = (const uint8_t*)duckdb_vector_get_data(v);
                    size_t elem_size = ctx->param_types[c]->size;
                    auto buf = std::make_unique<uint8_t[]>(selection.size() * elem_size);
                    for (size_t r = 0; r < selection.size(); ++r) {
                        std::memcpy(buf.get() + r * elem_size, src + selection[r] * elem_size,
                                    elem_size);
                    }
                    uint8_t* raw = buf.get();
                    nb::capsule owner(buf.release(),
                                      [](void* p) noexcept { delete[] (uint8_t*)p; });
                    args.append(nb::ndarray<nb::numpy, nb::ro>(raw, 1, shape, owner, nullptr,
                                                               ctx->param_types[c]->dtype()));
                }
                return nb::tuple(args);
            };

            // Fast path: all rows share one state (common for non-GROUP-BY queries).
            duckdb_aggregate_state first_state = states[all_rows ? 0 : rows[0]];
            bool single_state = true;
            for (idx_t i = 0; i < n && single_state; ++i) {
                single_state = states[i] == first_state;
            }
            if (single_state && all_rows) {
                // Owner-less arrays: nanobind hands Python a copy, which the
                // caller may keep.
                size_t shape[1] = {(size_t)n};
                nb::list args;
                for (idx_t c = 0; c < n_cols; ++c) {
                    duckdb_vector v = duckdb_data_chunk_get_vector(input, c);
                    void* data = duckdb_vector_get_data(v);
                    args.append(nb::ndarray<nb::numpy, nb::ro>(
                        data, 1, shape, nb::handle(), nullptr, ctx->param_types[c]->dtype()));
                }
                instance(ctx, first_state).attr("update")(*nb::tuple(args));
                return;
            }
            if (single_state) {
                instance(ctx, first_state).attr("update")(*gather_args(rows));
                return;
            }

            // Slow path: group row indices by state address.
            tsl::robin_map<duckdb_aggregate_state, std::vector<idx_t>> groups;
            groups.reserve(n);
            if (all_rows) {
                for (idx_t i = 0; i < n; ++i) groups[states[i]].push_back(i);
            } else {
                for (idx_t i : rows) groups[states[i]].push_back(i);
            }
            for (auto& [state, group_rows] : groups) {
                instance(ctx, state).attr("update")(*gather_args(group_rows));
            }
        },
        [&](const char* what) { duckdb_aggregate_function_set_error(info, what); });
}

void agg_combine(duckdb_function_info info, duckdb_aggregate_state* source,
                 duckdb_aggregate_state* target, idx_t count) {
    auto* ctx = (AggregateUDFContext*)duckdb_aggregate_function_get_extra_info(info);
    nb::gil_scoped_acquire gil;
    guard(
        [&] {
            for (idx_t i = 0; i < count; ++i) {
                // A source without rows has nothing to merge. Sources must stay
                // intact: window segment trees combine the same state into
                // several frames.
                auto* src = (PyObject*)source[i]->internal_ptr;
                if (!src) continue;
                instance(ctx, target[i]).attr("combine")(nb::handle(src));
            }
        },
        [&](const char* what) { duckdb_aggregate_function_set_error(info, what); });
}

void agg_finalize(duckdb_function_info info, duckdb_aggregate_state* source, duckdb_vector result,
                  idx_t count, idx_t offset) {
    auto* ctx = (AggregateUDFContext*)duckdb_aggregate_function_get_extra_info(info);
    nb::gil_scoped_acquire gil;

    uint8_t* out = (uint8_t*)duckdb_vector_get_data(result);
    size_t elem_size = ctx->return_type->size;
    nb::dlpack::dtype want = ctx->return_type->dtype();

    guard(
        [&] {
            for (idx_t i = 0; i < count; ++i) {
                // A state without rows finalizes a fresh instance (e.g. an
                // empty group or an aggregate over no rows).
                nb::object val = instance(ctx, source[i]).attr("finalize")();

                if (val.is_none()) {
                    duckdb_vector_ensure_validity_writable(result);
                    uint64_t* validity = duckdb_vector_get_validity(result);
                    duckdb_validity_set_row_invalid(validity, offset + i);
                    continue;
                }

                // Cast the returned scalar to the right C type and write it.
                uint8_t* slot = out + (offset + i) * elem_size;
                if (want.code == (uint8_t)nb::dlpack::dtype_code::Bool && want.bits == 8) {
                    // Truthiness, so numpy.bool and other bool-like scalars work.
                    int truth = PyObject_IsTrue(val.ptr());
                    if (truth < 0) throw nb::python_error();
                    *slot = truth ? 1 : 0;
                } else if (want.code == (uint8_t)nb::dlpack::dtype_code::Int) {
                    int64_t v = nb::cast<int64_t>(val);
                    // elem_size < 8 (TINYINT/SMALLINT/INTEGER/DATE): range-check
                    // before truncating, matching table.cpp's checked_signed.
                    if (elem_size < 8) {
                        int64_t bits = (int64_t)elem_size * 8;
                        int64_t lo = -(INT64_C(1) << (bits - 1));
                        int64_t hi = (INT64_C(1) << (bits - 1)) - 1;
                        if (v < lo || v > hi) {
                            throw DuckyError("ducky: finalize() value " + std::to_string(v) +
                                             " out of range for " + ctx->return_type->name);
                        }
                    }
                    std::memcpy(slot, &v, elem_size);
                } else if (want.code == (uint8_t)nb::dlpack::dtype_code::UInt) {
                    uint64_t v = nb::cast<uint64_t>(val);
                    if (elem_size < 8) {
                        uint64_t hi = (UINT64_C(1) << (elem_size * 8)) - 1;
                        if (v > hi) {
                            throw DuckyError("ducky: finalize() value " + std::to_string(v) +
                                             " out of range for " + ctx->return_type->name);
                        }
                    }
                    std::memcpy(slot, &v, elem_size);
                } else {
                    // Float / Double
                    double v = nb::cast<double>(val);
                    if (elem_size == 4) {
                        float f = (float)v;
                        std::memcpy(slot, &f, 4);
                    } else {
                        std::memcpy(slot, &v, 8);
                    }
                }
            }
        },
        [&](const char* what) { duckdb_aggregate_function_set_error(info, what); });
}

}  // namespace

void create_aggregate_function(Connection& con, const std::string& name, nb::object cls,
                               nb::object parameters, const std::string& return_type) {
    std::shared_ptr<DuckDBHandle> handle = con.open_handle();
    duckdb_connection raw = handle->connection;
    if (!nb::hasattr(cls, "combine")) {
        // DuckDB merges partial states not only across threads but also for
        // DISTINCT, window frames and large GROUP BYs, even single-threaded;
        // arbitrary Python state has no generic merge.
        throw DuckyError("ducky: aggregate UDF '" + name +
                         "' must define combine(self, other), which merges another partial state "
                         "into self; DuckDB merges states for parallel scans, DISTINCT, window "
                         "frames and large GROUP BYs");
    }
    auto ctx = std::make_unique<AggregateUDFContext>();
    ctx->cls = handle->keep(nb::cast<nb::callable>(cls));
    ctx->return_type = &lookup_typespec(return_type);

    if (nb::isinstance<nb::dict>(parameters)) {
        nb::dict d = nb::cast<nb::dict>(parameters);
        for (auto item : d) {
            ctx->param_types.push_back(&lookup_typespec(nb::cast<std::string>(item.second)));
        }
    } else {
        nb::sequence seq = nb::cast<nb::sequence>(parameters);
        for (auto item : seq) {
            ctx->param_types.push_back(&lookup_typespec(nb::cast<std::string>(item)));
        }
    }
    if (ctx->param_types.empty()) {
        throw DuckyError("ducky: aggregate UDF '" + name + "' must declare at least one parameter");
    }

    duckdb_aggregate_function f = duckdb_create_aggregate_function();
    duckdb_aggregate_function_set_name(f, name.c_str());

    for (const TypeSpec* t : ctx->param_types) {
        duckdb_logical_type lt = duckdb_create_logical_type(t->type);
        duckdb_aggregate_function_add_parameter(f, lt);
        duckdb_destroy_logical_type(&lt);
    }
    {
        duckdb_logical_type rt = duckdb_create_logical_type(ctx->return_type->type);
        duckdb_aggregate_function_set_return_type(f, rt);
        duckdb_destroy_logical_type(&rt);
    }

    duckdb_aggregate_function_set_functions(f, &agg_state_size, &agg_init, &agg_update,
                                            &agg_combine, &agg_finalize);
    duckdb_aggregate_function_set_destructor(f, &agg_destroy);
    // Ownership of ctx passes to the function object here: its destroy callback
    // runs whether or not registration succeeds (a failed registration's
    // duckdb_destroy_aggregate_function drops the last reference).
    duckdb_aggregate_function_set_extra_info(f, ctx.get(), &gil_delete<AggregateUDFContext>);
    // NOLINTNEXTLINE(bugprone-unused-return-value)
    (void)ctx.release();

    duckdb_state state;
    {
        // Registration takes the client context lock, which a query running
        // Python callbacks on another thread may hold while it waits for the
        // GIL; a failed registration's destructor takes the GIL itself.
        release_gil_if_held release;
        state = duckdb_register_aggregate_function(raw, f);
        duckdb_destroy_aggregate_function(&f);
    }
    if (state == DuckDBError) {
        throw DuckyError("ducky: failed to register aggregate function '" + name + "'");
    }
}
