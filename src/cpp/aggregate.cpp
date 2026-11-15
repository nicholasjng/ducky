#include "aggregate.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <tsl/robin_map.h>

#include <cstring>
#include <memory>
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
using AggState = void*;

PyObject*& slot(AggState state) { return *(PyObject**)state; }

void agg_state_size(duckdb_v2_aggregate_function_size_info_handle info,
                    duckdb_v2_error_info_handle* err) {
    report(err, [&] {
        V2Error e;
        e.check(duckdb_v2_aggregate_function_size_set_state_size(info, sizeof(PyObject*), e.out()));
    });
}

void agg_init(duckdb_v2_aggregate_function_init_info_handle info,
              duckdb_v2_error_info_handle* err) {
    report(err, [&] {
        V2Error e;
        AggState* states = nullptr;
        idx_t count = 0;
        e.check(duckdb_v2_aggregate_function_init_get_states(info, &states, e.out()));
        e.check(duckdb_v2_aggregate_function_init_get_state_count(info, &count, e.out()));
        // State memory isn't zero-initialized by DuckDB, and agg_destroy
        // unconditionally Py_XDECREFs the slot.
        for (idx_t i = 0; i < count; ++i) slot(states[i]) = nullptr;
    });
}

// The instance in `state`, created on first use.
nb::handle instance(AggregateUDFContext* ctx, AggState state) {
    if (!slot(state)) slot(state) = ctx->cls().release().ptr();
    return nb::handle(slot(state));
}

void agg_destroy(duckdb_v2_aggregate_function_destroy_info_handle info,
                 duckdb_v2_error_info_handle* err) {
    AggState* states = nullptr;
    idx_t count = 0;
    if (duckdb_v2_aggregate_function_destroy_get_states(info, &states, nullptr) !=
            DUCKDB_V2_ERROR_NONE ||
        duckdb_v2_aggregate_function_destroy_get_state_count(info, &count, nullptr) !=
            DUCKDB_V2_ERROR_NONE) {
        return;
    }
    nb::gil_scoped_acquire gil;
    for (idx_t i = 0; i < count; ++i) {
        Py_XDECREF(slot(states[i]));
        slot(states[i]) = nullptr;
    }
}

bool row_is_valid(const uint64_t* validity, idx_t row) {
    return !validity || ((validity[row / 64] >> (row % 64)) & 1);
}

// The function's context, from any callback's info.
template <typename Info, typename Getter>
AggregateUDFContext* context_of(Info info, Getter getter) {
    V2Error e;
    void* user_data = nullptr;
    e.check(getter(info, &user_data, e.out()));
    return (AggregateUDFContext*)user_data;
}

void agg_update(duckdb_v2_aggregate_function_update_info_handle info,
                duckdb_v2_error_info_handle* err) {
    nb::gil_scoped_acquire gil;
    report(err, [&] {
        V2Error e;
        auto* ctx = context_of(info, &duckdb_v2_aggregate_function_update_get_user_data);
        idx_t n = 0;
        e.check(duckdb_v2_aggregate_function_update_get_row_count(info, &n, e.out()));
        if (n == 0) return;
        AggState* states = nullptr;
        e.check(duckdb_v2_aggregate_function_update_get_states(info, &states, e.out()));
        idx_t n_cols = (idx_t)ctx->param_types.size();
        // Flat arguments, which v2 hands over as they come.
        std::vector<duckdb_v2_vector_view> views(n_cols);
        for (idx_t c = 0; c < n_cols; ++c) {
            duckdb_v2_vector_handle arg = nullptr;
            e.check(duckdb_v2_aggregate_function_update_get_arg(info, c, &arg, e.out()));
            e.check(duckdb_v2_vector_flatten(arg, e.out()));
            e.check(duckdb_v2_vector_get_view(arg, &views[c], e.out()));
        }

        // SQL aggregates ignore rows with a NULL argument; DuckDB passes
        // them in, so leave them out here. `rows` stays empty when every
        // row is valid.
        std::vector<const uint64_t*> masks;
        for (const duckdb_v2_vector_view& view : views) {
            if (view.validity) masks.push_back(view.validity);
        }
        std::vector<idx_t> rows;
        if (!masks.empty()) {
            rows.reserve(n);
            for (idx_t r = 0; r < n; ++r) {
                bool valid = true;
                for (const uint64_t* m : masks) {
                    if (!row_is_valid(m, r)) {
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

        // Input arrays for the rows `selection` lists, gathered into
        // owned buffers (the caller may keep the arrays).
        auto gather_args = [&](const std::vector<idx_t>& selection) {
            size_t shape[1] = {selection.size()};
            nb::list args;
            for (idx_t c = 0; c < n_cols; ++c) {
                const uint8_t* src = (const uint8_t*)views[c].data;
                size_t elem_size = ctx->param_types[c]->size;
                auto buf = std::make_unique<uint8_t[]>(selection.size() * elem_size);
                for (size_t r = 0; r < selection.size(); ++r) {
                    std::memcpy(buf.get() + r * elem_size, src + selection[r] * elem_size,
                                elem_size);
                }
                uint8_t* raw = buf.get();
                nb::capsule owner(buf.release(), [](void* p) noexcept { delete[] (uint8_t*)p; });
                args.append(nb::ndarray<nb::numpy, nb::ro>(raw, 1, shape, owner, nullptr,
                                                           ctx->param_types[c]->dtype()));
            }
            return nb::tuple(args);
        };

        // Fast path: all rows share one state (common for non-GROUP-BY queries).
        AggState first_state = states[all_rows ? 0 : rows[0]];
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
                void* data = (void*)views[c].data;
                args.append(nb::ndarray<nb::numpy, nb::ro>(data, 1, shape, nb::handle(), nullptr,
                                                           ctx->param_types[c]->dtype()));
            }
            instance(ctx, first_state).attr("update")(*nb::tuple(args));
            return;
        }
        if (single_state) {
            instance(ctx, first_state).attr("update")(*gather_args(rows));
            return;
        }

        // Slow path: group row indices by state address.
        tsl::robin_map<AggState, std::vector<idx_t>> groups;
        groups.reserve(n);
        if (all_rows) {
            for (idx_t i = 0; i < n; ++i) groups[states[i]].push_back(i);
        } else {
            for (idx_t i : rows) groups[states[i]].push_back(i);
        }
        for (auto& [state, group_rows] : groups) {
            instance(ctx, state).attr("update")(*gather_args(group_rows));
        }
    });
}

void agg_combine(duckdb_v2_aggregate_function_combine_info_handle info,
                 duckdb_v2_error_info_handle* err) {
    nb::gil_scoped_acquire gil;
    report(err, [&] {
        V2Error e;
        auto* ctx = context_of(info, &duckdb_v2_aggregate_function_combine_get_user_data);
        AggState* source = nullptr;
        AggState* target = nullptr;
        idx_t count = 0;
        e.check(duckdb_v2_aggregate_function_combine_get_sources(info, &source, e.out()));
        e.check(duckdb_v2_aggregate_function_combine_get_targets(info, &target, e.out()));
        e.check(duckdb_v2_aggregate_function_combine_get_state_count(info, &count, e.out()));
        for (idx_t i = 0; i < count; ++i) {
            // A source without rows has nothing to merge. Sources must stay
            // intact: window segment trees combine the same state into
            // several frames.
            PyObject* src = slot(source[i]);
            if (!src) continue;
            instance(ctx, target[i]).attr("combine")(nb::handle(src));
        }
    });
}

void agg_finalize(duckdb_v2_aggregate_function_finalize_info_handle info,
                  duckdb_v2_error_info_handle* err) {
    nb::gil_scoped_acquire gil;
    report(err, [&] {
        V2Error e;
        auto* ctx = context_of(info, &duckdb_v2_aggregate_function_finalize_get_user_data);
        AggState* source = nullptr;
        idx_t count = 0, offset = 0;
        duckdb_v2_vector_handle result = nullptr;
        e.check(duckdb_v2_aggregate_function_finalize_get_states(info, &source, e.out()));
        e.check(duckdb_v2_aggregate_function_finalize_get_state_count(info, &count, e.out()));
        e.check(duckdb_v2_aggregate_function_finalize_get_result(info, &result, e.out()));
        e.check(duckdb_v2_aggregate_function_finalize_get_result_offset(info, &offset, e.out()));
        void* data = nullptr;
        e.check(duckdb_v2_vector_get_data_mutable(result, &data, e.out()));
        uint8_t* out = (uint8_t*)data;
        uint64_t* validity = nullptr;
        size_t elem_size = ctx->return_type->size;
        nb::dlpack::dtype want = ctx->return_type->dtype();

        for (idx_t i = 0; i < count; ++i) {
            // A state without rows finalizes a fresh instance (e.g. an
            // empty group or an aggregate over no rows).
            nb::object val = instance(ctx, source[i]).attr("finalize")();

            if (val.is_none()) {
                if (!validity) {
                    e.check(duckdb_v2_vector_flat_get_validity_mutable(result, &validity, e.out()));
                }
                idx_t row = offset + i;
                validity[row / 64] &= ~((uint64_t)1 << (row % 64));
                continue;
            }

            // Cast the returned scalar to the right C type and write it.
            uint8_t* cell = out + (offset + i) * elem_size;
            if (want.code == (uint8_t)nb::dlpack::dtype_code::Bool && want.bits == 8) {
                // Truthiness, so numpy.bool and other bool-like scalars work.
                int truth = PyObject_IsTrue(val.ptr());
                if (truth < 0) throw nb::python_error();
                *cell = truth ? 1 : 0;
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
                std::memcpy(cell, &v, elem_size);
            } else if (want.code == (uint8_t)nb::dlpack::dtype_code::UInt) {
                uint64_t v = nb::cast<uint64_t>(val);
                if (elem_size < 8) {
                    uint64_t hi = (UINT64_C(1) << (elem_size * 8)) - 1;
                    if (v > hi) {
                        throw DuckyError("ducky: finalize() value " + std::to_string(v) +
                                         " out of range for " + ctx->return_type->name);
                    }
                }
                std::memcpy(cell, &v, elem_size);
            } else {
                // Float / Double
                double v = nb::cast<double>(val);
                if (elem_size == 4) {
                    float f = (float)v;
                    std::memcpy(cell, &f, 4);
                } else {
                    std::memcpy(cell, &v, 8);
                }
            }
        }
    });
}

}  // namespace

void create_aggregate_function(Connection& con, const std::string& name, nb::object cls,
                               nb::object parameters, const std::string& return_type) {
    std::shared_ptr<DuckDBHandle> handle = con.open_handle();
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

    const std::string failed = "ducky: failed to register aggregate function '" + name + "': ";
    V2Error err;
    auto check = [&](DUCKDB_V2_ERROR code) { err.check(code, failed); };
    duckdb_v2_factory_handle factory = nullptr;
    check(duckdb_v2_connection_get_factory(handle->connection_v2, &factory, err.out()));
    duckdb_v2_aggregate_function_handle f = nullptr;
    check(duckdb_v2_aggregate_function_create(factory, &f, err.out()));
    std::unique_ptr<duckdb_v2_aggregate_function_handle,
                    void (*)(duckdb_v2_aggregate_function_handle*)>
        f_guard(&f, [](duckdb_v2_aggregate_function_handle* p) {
            // Destroying an unregistered function drops the last reference to
            // its user data, whose destructor takes the GIL.
            release_gil_if_held release;
            duckdb_v2_aggregate_function_destroy(p);
        });
    check(duckdb_v2_aggregate_function_set_name(f, name.data(), name.size(), err.out()));

    // Adds a parameter or sets the return type to a type id, through `add`.
    auto with_type = [&](DUCKDB_V2_LOGICAL_TYPE_ID id, auto add) {
        duckdb_v2_logical_type_handle t = nullptr;
        check(duckdb_v2_factory_create_type_from_id(factory, id, nullptr, nullptr, nullptr, 0, &t,
                                                    err.out()));
        DUCKDB_V2_ERROR code = add(t);
        duckdb_v2_logical_type_destroy(&t);
        check(code);
    };
    duckdb_v2_function_signature_handle sig = nullptr;
    check(duckdb_v2_aggregate_function_get_signature(f, &sig, err.out()));
    for (size_t i = 0; i < ctx->param_types.size(); ++i) {
        std::string param = "arg" + std::to_string(i + 1);
        with_type(ctx->param_types[i]->v2_type, [&](duckdb_v2_logical_type_handle t) {
            return duckdb_v2_function_signature_add_parameter(
                sig, param.data(), param.size(), t, nullptr,
                DUCKDB_V2_FUNCTION_PARAMETER_KIND_POSITIONAL_ONLY, err.out());
        });
    }
    with_type(ctx->return_type->v2_type, [&](duckdb_v2_logical_type_handle t) {
        return duckdb_v2_function_signature_set_return_type(sig, t, err.out());
    });

    check(duckdb_v2_aggregate_function_set_size_callback(f, &agg_state_size, err.out()));
    check(duckdb_v2_aggregate_function_set_init_callback(f, &agg_init, err.out()));
    check(duckdb_v2_aggregate_function_set_update_callback(f, &agg_update, err.out()));
    check(duckdb_v2_aggregate_function_set_combine_callback(f, &agg_combine, err.out()));
    check(duckdb_v2_aggregate_function_set_finalize_callback(f, &agg_finalize, err.out()));
    check(duckdb_v2_aggregate_function_set_destroy_callback(f, &agg_destroy, err.out()));
    // Ownership of ctx passes to the function object here: its destroy callback
    // runs whether or not registration succeeds.
    duckdb_v2_opaque data{ctx.get(), &gil_delete<AggregateUDFContext>, nullptr};
    check(duckdb_v2_aggregate_function_set_user_data(f, &data, err.out()));
    // NOLINTNEXTLINE(bugprone-unused-return-value)
    (void)ctx.release();

    DUCKDB_V2_ERROR code;
    {
        // Registration takes the client context lock, which a query running
        // Python callbacks on another thread may hold while it waits for the
        // GIL.
        release_gil_if_held release;
        code =
            duckdb_v2_connection_register_aggregate_function(handle->connection_v2, f, err.out());
    }
    check(code);
}
