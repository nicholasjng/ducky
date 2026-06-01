#include "aggregate.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <unordered_map>
#include <vector>

#include "connection.hpp"
#include "ducky.hpp"
#include "function.hpp"

namespace nb = nanobind;

namespace {

void agg_extra_info_destroy(void* ptr) {
    if (!ptr) return;
    nb::gil_scoped_acquire gil;
    delete (AggregateUDFContext*)ptr;
}

idx_t agg_state_size(duckdb_function_info /*info*/) { return sizeof(PyObject*); }

void agg_init(duckdb_function_info info, duckdb_aggregate_state state) {
    auto* ctx = (AggregateUDFContext*)duckdb_aggregate_function_get_extra_info(info);
    nb::gil_scoped_acquire gil;
    try {
        nb::object inst = ctx->cls();
        PyObject* raw = inst.release().ptr();
        state->internal_ptr = (void*)raw;
    } catch (nb::python_error& e) {
        duckdb_aggregate_function_set_error(info, e.what());
    } catch (const std::exception& e) {
        duckdb_aggregate_function_set_error(info, e.what());
    }
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
    nb::gil_scoped_acquire gil;

    idx_t n = duckdb_data_chunk_get_size(input);
    if (n == 0) return;

    idx_t n_cols = (idx_t)ctx->param_types.size();

    try {
        // Fast path: all rows share one state (common for non-GROUP-BY queries).
        bool single_state = true;
        duckdb_aggregate_state first_state = states[0];
        for (idx_t i = 1; i < n; ++i) {
            if (states[i] != first_state) {
                single_state = false;
                break;
            }
        }

        if (single_state) {
            size_t shape[1] = {(size_t)n};
            nb::list args;
            for (idx_t c = 0; c < n_cols; ++c) {
                duckdb_vector v = duckdb_data_chunk_get_vector(input, c);
                void* data = duckdb_vector_get_data(v);
                args.append(nb::ndarray<nb::numpy, nb::ro>(data, 1, shape, nb::handle(), nullptr,
                                                           ctx->param_types[c]->dtype()));
            }
            nb::object inst = nb::borrow((PyObject*)first_state->internal_ptr);
            inst.attr("update")(*nb::tuple(args));
            return;
        }

        // Slow path: group row indices by state address.
        std::unordered_map<duckdb_aggregate_state, std::vector<idx_t>> groups;
        groups.reserve(n);
        for (idx_t i = 0; i < n; ++i) {
            groups[states[i]].push_back(i);
        }

        for (auto& [state, rows] : groups) {
            idx_t group_n = (idx_t)rows.size();
            size_t shape[1] = {(size_t)group_n};
            nb::list args;
            for (idx_t c = 0; c < n_cols; ++c) {
                duckdb_vector v = duckdb_data_chunk_get_vector(input, c);
                const uint8_t* src = (const uint8_t*)duckdb_vector_get_data(v);
                size_t elem_size = ctx->param_types[c]->size;
                // Gather non-contiguous rows into a contiguous buffer.
                auto buf = std::make_unique<uint8_t[]>(group_n * elem_size);
                for (idx_t r = 0; r < group_n; ++r) {
                    std::memcpy(buf.get() + r * elem_size, src + rows[r] * elem_size, elem_size);
                }
                uint8_t* raw = buf.get();
                nb::capsule owner(buf.release(), [](void* p) noexcept { delete[] (uint8_t*)p; });
                args.append(nb::ndarray<nb::numpy, nb::ro>(raw, 1, shape, owner, nullptr,
                                                           ctx->param_types[c]->dtype()));
            }
            nb::object inst = nb::borrow((PyObject*)state->internal_ptr);
            inst.attr("update")(*nb::tuple(args));
        }
    } catch (nb::python_error& e) {
        duckdb_aggregate_function_set_error(info, e.what());
    } catch (const std::exception& e) {
        duckdb_aggregate_function_set_error(info, e.what());
    }
}

// Fallback for aggregates without a Python combine(): copies source's __dict__
// into target so DuckDB's finalize sees the accumulated state.  Correct for
// serial execution; may produce wrong results under parallel GROUP BY merges.
void agg_combine_copy(duckdb_function_info info, duckdb_aggregate_state* source,
                      duckdb_aggregate_state* target, idx_t count) {
    nb::gil_scoped_acquire gil;
    try {
        for (idx_t i = 0; i < count; ++i) {
            nb::object src_inst = nb::borrow((PyObject*)source[i]->internal_ptr);
            nb::object tgt_inst = nb::borrow((PyObject*)target[i]->internal_ptr);
            tgt_inst.attr("__dict__").attr("update")(src_inst.attr("__dict__"));
        }
    } catch (nb::python_error& e) {
        duckdb_aggregate_function_set_error(info, e.what());
    } catch (const std::exception& e) {
        duckdb_aggregate_function_set_error(info, e.what());
    }
}

void agg_combine(duckdb_function_info info, duckdb_aggregate_state* source,
                 duckdb_aggregate_state* target, idx_t count) {
    nb::gil_scoped_acquire gil;
    try {
        for (idx_t i = 0; i < count; ++i) {
            nb::object src_inst = nb::borrow((PyObject*)source[i]->internal_ptr);
            nb::object tgt_inst = nb::borrow((PyObject*)target[i]->internal_ptr);
            tgt_inst.attr("combine")(src_inst);
        }
    } catch (nb::python_error& e) {
        duckdb_aggregate_function_set_error(info, e.what());
    } catch (const std::exception& e) {
        duckdb_aggregate_function_set_error(info, e.what());
    }
}

void agg_finalize(duckdb_function_info info, duckdb_aggregate_state* source, duckdb_vector result,
                  idx_t count, idx_t offset) {
    auto* ctx = (AggregateUDFContext*)duckdb_aggregate_function_get_extra_info(info);
    nb::gil_scoped_acquire gil;

    uint8_t* out = (uint8_t*)duckdb_vector_get_data(result);
    size_t elem_size = ctx->return_type->size;
    nb::dlpack::dtype want = ctx->return_type->dtype();

    try {
        for (idx_t i = 0; i < count; ++i) {
            nb::object inst = nb::borrow((PyObject*)source[i]->internal_ptr);
            nb::object val = inst.attr("finalize")();

            if (val.is_none()) {
                duckdb_vector_ensure_validity_writable(result);
                uint64_t* validity = duckdb_vector_get_validity(result);
                duckdb_validity_set_row_invalid(validity, offset + i);
                continue;
            }

            // Cast the returned scalar to the right C type and write it.
            uint8_t* slot = out + (offset + i) * elem_size;
            if (want.code == (uint8_t)nb::dlpack::dtype_code::Bool && want.bits == 8) {
                bool v = nb::cast<bool>(val);
                *slot = v ? 1 : 0;
            } else if (want.code == (uint8_t)nb::dlpack::dtype_code::Int) {
                int64_t v = nb::cast<int64_t>(val);
                std::memcpy(slot, &v, elem_size);
            } else if (want.code == (uint8_t)nb::dlpack::dtype_code::UInt) {
                uint64_t v = nb::cast<uint64_t>(val);
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
    } catch (nb::python_error& e) {
        duckdb_aggregate_function_set_error(info, e.what());
    } catch (const std::exception& e) {
        duckdb_aggregate_function_set_error(info, e.what());
    }
}

}  // namespace

void create_aggregate_function(Connection& con, const std::string& name, nb::object cls,
                               nb::object parameters, const std::string& return_type) {
    auto ctx = std::make_unique<AggregateUDFContext>();
    ctx->cls = nb::cast<nb::callable>(cls);
    ctx->return_type = &lookup_typespec(return_type);
    ctx->has_combine = nb::hasattr(cls, "combine");

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
                                            ctx->has_combine ? &agg_combine : &agg_combine_copy,
                                            &agg_finalize);
    duckdb_aggregate_function_set_destructor(f, &agg_destroy);
    duckdb_aggregate_function_set_extra_info(f, ctx.get(), &agg_extra_info_destroy);

    duckdb_connection raw = con.raw_connection();
    duckdb_state state = duckdb_register_aggregate_function(raw, f);
    duckdb_destroy_aggregate_function(&f);
    if (state == DuckDBError) {
        throw DuckyError("ducky: failed to register aggregate function '" + name + "'");
    }
    // NOLINTNEXTLINE(bugprone-unused-return-value)
    (void)ctx.release();
}
