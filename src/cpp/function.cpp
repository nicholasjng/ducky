#include "function.hpp"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "arrow_abi.h"
#include "arrow_util.hpp"
#include "connection.hpp"
#include "ducky.hpp"

namespace {

using namespace nb::literals;

template <typename T>
nb::dlpack::dtype dt() {
    return nb::dtype<T>();
}

// Cached pyarrow callables hit on every Arrow-UDF chunk.
struct PyArrowAPI {
    nb::object import_from_c;  // pa.RecordBatch._import_from_c
    nb::object from_arrays;    // pa.RecordBatch.from_arrays
    nb::object schema_import;  // pa.Schema._import_from_c
};

const PyArrowAPI& pyarrow_api() {
    static std::atomic<PyArrowAPI*> cached{nullptr};
    static nb::ft_mutex mu;
    return cached_singleton(cached, mu, [] {
        nb::module_ pa = nb::module_::import_("pyarrow");
        nb::object rb_cls = pa.attr("RecordBatch");
        return PyArrowAPI{rb_cls.attr("_import_from_c"), rb_cls.attr("from_arrays"),
                          pa.attr("Schema").attr("_import_from_c")};
    });
}

// Matches the set Chunk.column() exposes, so UDF inputs and chunk columns agree.
const TypeSpec kTypes[] = {
    {"BOOLEAN", DUCKDB_TYPE_BOOLEAN, DUCKDB_V2_LOGICAL_TYPE_ID_BOOLEAN, sizeof(bool), &dt<bool>},
    {"TINYINT", DUCKDB_TYPE_TINYINT, DUCKDB_V2_LOGICAL_TYPE_ID_TINYINT, sizeof(int8_t),
     &dt<int8_t>},
    {"SMALLINT", DUCKDB_TYPE_SMALLINT, DUCKDB_V2_LOGICAL_TYPE_ID_SMALLINT, sizeof(int16_t),
     &dt<int16_t>},
    {"INTEGER", DUCKDB_TYPE_INTEGER, DUCKDB_V2_LOGICAL_TYPE_ID_INTEGER, sizeof(int32_t),
     &dt<int32_t>},
    {"BIGINT", DUCKDB_TYPE_BIGINT, DUCKDB_V2_LOGICAL_TYPE_ID_BIGINT, sizeof(int64_t), &dt<int64_t>},
    {"UTINYINT", DUCKDB_TYPE_UTINYINT, DUCKDB_V2_LOGICAL_TYPE_ID_UTINYINT, sizeof(uint8_t),
     &dt<uint8_t>},
    {"USMALLINT", DUCKDB_TYPE_USMALLINT, DUCKDB_V2_LOGICAL_TYPE_ID_USMALLINT, sizeof(uint16_t),
     &dt<uint16_t>},
    {"UINTEGER", DUCKDB_TYPE_UINTEGER, DUCKDB_V2_LOGICAL_TYPE_ID_UINTEGER, sizeof(uint32_t),
     &dt<uint32_t>},
    {"UBIGINT", DUCKDB_TYPE_UBIGINT, DUCKDB_V2_LOGICAL_TYPE_ID_UBIGINT, sizeof(uint64_t),
     &dt<uint64_t>},
    {"FLOAT", DUCKDB_TYPE_FLOAT, DUCKDB_V2_LOGICAL_TYPE_ID_FLOAT, sizeof(float), &dt<float>},
    {"DOUBLE", DUCKDB_TYPE_DOUBLE, DUCKDB_V2_LOGICAL_TYPE_ID_DOUBLE, sizeof(double), &dt<double>},
    {"DATE", DUCKDB_TYPE_DATE, DUCKDB_V2_LOGICAL_TYPE_ID_DATE, sizeof(int32_t), &dt<int32_t>},
    {"TIME", DUCKDB_TYPE_TIME, DUCKDB_V2_LOGICAL_TYPE_ID_TIME, sizeof(int64_t), &dt<int64_t>},
    {"TIMESTAMP", DUCKDB_TYPE_TIMESTAMP, DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP, sizeof(int64_t),
     &dt<int64_t>},
    {"TIMESTAMP_S", DUCKDB_TYPE_TIMESTAMP_S, DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_SEC,
     sizeof(int64_t), &dt<int64_t>},
    {"TIMESTAMP_MS", DUCKDB_TYPE_TIMESTAMP_MS, DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_MS,
     sizeof(int64_t), &dt<int64_t>},
    {"TIMESTAMP_NS", DUCKDB_TYPE_TIMESTAMP_NS, DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_NS,
     sizeof(int64_t), &dt<int64_t>},
    {"TIMESTAMP_TZ", DUCKDB_TYPE_TIMESTAMP_TZ, DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ,
     sizeof(int64_t), &dt<int64_t>},
};

const TypeSpec& lookup(const std::string& name) {
    for (const TypeSpec& t : kTypes) {
        if (name == t.name) return t;
    }
    throw DuckyError("ducky: unknown UDF type '" + name +
                     "' (supported: BOOLEAN, [U]TINYINT..BIGINT, FLOAT, DOUBLE, "
                     "DATE, TIME, TIMESTAMP[_S/MS/NS/TZ])");
}

// Only bool, int and float map unambiguously, so anything else must be passed
// as `parameters=` or `return_type=`.
std::string annotation_to_type(nb::handle hint, const std::string& where) {
    nb::module_ builtins = nb::module_::import_("builtins");
    if (hint.is(builtins.attr("bool"))) return "BOOLEAN";
    if (hint.is(builtins.attr("int"))) return "BIGINT";
    if (hint.is(builtins.attr("float"))) return "DOUBLE";
    std::string repr = nb::cast<std::string>(nb::str(hint));
    throw DuckyError("ducky: cannot infer DuckDB type from annotation " + repr + " on " + where +
                     " (supported: bool, int, float — pass `parameters=`/`return_type=` "
                     "explicitly for other types)");
}

nb::list infer_parameters(nb::callable fn) {
    nb::module_ inspect = nb::module_::import_("inspect");
    nb::module_ typing = nb::module_::import_("typing");
    nb::object sig = inspect.attr("signature")(fn);
    nb::object params = sig.attr("parameters");  // ordered mapping
    nb::dict hints;
    try {
        hints = nb::cast<nb::dict>(typing.attr("get_type_hints")(fn));
    } catch (nb::python_error&) {
        throw DuckyError("ducky: failed to resolve type hints on UDF for inference");
    }
    nb::list out;
    for (auto item : params) {
        std::string pname = nb::cast<std::string>(item);
        if (!hints.contains(pname.c_str())) {
            throw DuckyError("ducky: UDF parameter '" + pname +
                             "' has no type annotation (annotate it, or pass `parameters=` "
                             "explicitly)");
        }
        out.append(annotation_to_type(hints[pname.c_str()], "parameter '" + pname + "'"));
    }
    return out;
}

std::string infer_return_type(nb::callable fn) {
    nb::module_ typing = nb::module_::import_("typing");
    nb::dict hints;
    try {
        hints = nb::cast<nb::dict>(typing.attr("get_type_hints")(fn));
    } catch (nb::python_error&) {
        throw DuckyError("ducky: failed to resolve type hints on UDF for inference");
    }
    if (!hints.contains("return")) {
        throw DuckyError(
            "ducky: UDF has no return type annotation (annotate it, or pass "
            "`return_type=` explicitly)");
    }
    return annotation_to_type(hints["return"], "return value");
}

// Exceptions must not cross into DuckDB, so they are reported through `err`.
template <typename Body>
void report_udf_error(duckdb_v2_error_info_handle* err, const char* prefix, Body&& body) {
    report(err, [&] {
        try {
            body();
        } catch (nb::python_error& e) {
            throw DuckyError(prefix + std::string(e.what()));
        } catch (const std::exception& e) {
            throw DuckyError(prefix + std::string(e.what()));
        }
    });
}

bool row_is_valid(const uint64_t* validity, idx_t row) {
    return !validity || ((validity[row / 64] >> (row % 64)) & 1);
}

// Takes the validity mask on first use, since most outputs have no NULLs.
void set_row_null(duckdb_v2_vector_handle vector, uint64_t*& validity, idx_t row) {
    if (!validity) {
        V2Error e;
        e.check(duckdb_v2_vector_flat_get_validity_mutable(vector, &validity, e.out()));
    }
    validity[row / 64] &= ~((uint64_t)1 << (row % 64));
}

// DuckDB hands the trampoline only its extra_info slot, so everything lives here.
struct UDFContext {
    // Kept alive by the DuckDBHandle.
    nb::handle callable;
    nb::handle init_callable;  // optional per-worker-thread state factory
    // Non-empty: the callable receives one dict {name: ndarray}. Empty: it
    // receives the ndarrays as positional arguments.
    std::vector<std::string> param_names;
    std::vector<const TypeSpec*> param_types;
    const TypeSpec* return_type;
    // Set for varargs, all of this type, in which case `param_types` is ignored
    // and the arity comes from the input chunk.
    const TypeSpec* varargs_type = nullptr;
    // NULL-aware: each input arg is delivered as a (values, mask) tuple, where
    // `mask` is a 1-D uint8 ndarray (1=valid, 0=NULL). Output may emit NULLs
    // via the same (values, mask) return convention.
    bool special_handling = false;
};

// Freed under the GIL at thread teardown, since it holds a Python object.
struct UDFState {
    nb::object obj;
};

void udf_init_trampoline(duckdb_v2_scalar_function_init_info_handle info,
                         duckdb_v2_context_handle context, duckdb_v2_error_info_handle* err) {
    nb::gil_scoped_acquire gil;
    report_udf_error(err, "ducky UDF init error: ", [&] {
        V2Error e;
        void* user_data = nullptr;
        e.check(duckdb_v2_scalar_function_init_get_user_data(info, &user_data, e.out()));
        auto* ctx = (UDFContext*)user_data;
        auto holder = std::make_unique<UDFState>(UDFState{ctx->init_callable()});
        duckdb_v2_opaque data{holder.get(), &gil_delete<UDFState>, nullptr};
        e.check(duckdb_v2_scalar_function_init_set_init_data(info, &data, e.out()));
        // NOLINTNEXTLINE(bugprone-unused-return-value)
        (void)holder.release();
    });
}

// Arrays built over the buffer with `owner` reach Python without a copy.
char* owned_buffer(size_t bytes, nb::object& owner) {
    char* buf = new char[bytes ? bytes : 1];
    owner = nb::capsule(buf, [](void* p) noexcept { delete[] (char*)p; });
    return buf;
}

// 1 marks a valid row, 0 a NULL.
nb::object build_input_mask(const uint64_t* validity, idx_t n) {
    nb::object owner;
    uint8_t* mask = (uint8_t*)owned_buffer(n, owner);
    if (!validity) {
        std::memset(mask, 1, n);
    } else {
        for (idx_t i = 0; i < n; ++i) mask[i] = row_is_valid(validity, i) ? 1 : 0;
    }
    size_t shape[1] = {(size_t)n};
    return nb::cast(
        nb::ndarray<nb::numpy, nb::ro>(mask, 1, shape, owner, nullptr, nb::dtype<uint8_t>()));
}

// DuckDB only short-circuits constant NULL inputs, so a UDF without special
// handling would see NULL rows of a flat chunk too. False when the first `n`
// rows have no NULL, so the caller can pass the chunk as is.
bool collect_valid_rows(const std::vector<duckdb_v2_vector_view>& args, idx_t n,
                        std::vector<idx_t>& keep) {
    std::vector<const uint64_t*> masks;
    for (const duckdb_v2_vector_view& arg : args) {
        if (arg.validity) masks.push_back(arg.validity);
    }
    if (masks.empty()) return false;
    keep.clear();
    keep.reserve(n);
    for (idx_t r = 0; r < n; ++r) {
        bool valid = true;
        for (const uint64_t* m : masks) {
            if (!row_is_valid(m, r)) {
                valid = false;
                break;
            }
        }
        if (valid) keep.push_back(r);
    }
    return keep.size() != n;
}

void udf_trampoline(duckdb_v2_scalar_function_exec_info_handle info,
                    duckdb_v2_context_handle context, duckdb_v2_error_info_handle* err) {
    nb::gil_scoped_acquire gil;
    report_udf_error(err, "ducky UDF error: ", [&] {
        V2Error e;
        void* user_data = nullptr;
        e.check(duckdb_v2_scalar_function_exec_get_user_data(info, &user_data, e.out()));
        auto* ctx = (UDFContext*)user_data;
        idx_t n = 0;
        e.check(duckdb_v2_scalar_function_exec_get_row_count(info, &n, e.out()));
        idx_t n_params = ctx->param_types.size();
        if (ctx->varargs_type) {
            e.check(duckdb_v2_scalar_function_exec_get_arg_count(info, nullptr, &n_params, nullptr,
                                                                 nullptr, e.out()));
        }
        // Flat arguments, which v2 hands over as they come.
        std::vector<duckdb_v2_vector_view> args(n_params);
        for (idx_t i = 0; i < n_params; ++i) {
            duckdb_v2_vector_handle arg = nullptr;
            e.check(duckdb_v2_scalar_function_exec_get_arg(info, i, &arg, e.out()));
            e.check(duckdb_v2_vector_flatten(arg, e.out()));
            e.check(duckdb_v2_vector_get_view(arg, &args[i], e.out()));
        }
        duckdb_v2_vector_handle output = nullptr;
        e.check(duckdb_v2_scalar_function_exec_get_result(info, &output, e.out()));
        uint64_t* out_validity = nullptr;

        // Without special handling, rows with a NULL input are left out of
        // the call (their output is NULL), so the callable never sees the
        // arbitrary values DuckDB leaves in NULL slots.
        std::vector<idx_t> keep;
        bool compact = !ctx->special_handling && collect_valid_rows(args, n, keep);
        idx_t m = compact ? (idx_t)keep.size() : n;
        size_t shape[1] = {(size_t)m};

        if (compact) {
            size_t k = 0;
            for (idx_t r = 0; r < n; ++r) {
                if (k < keep.size() && keep[k] == r) {
                    ++k;
                } else {
                    set_row_null(output, out_validity, r);
                }
            }
            if (m == 0) return;  // every row has a NULL input
        }

        // Input arrays own their data, so the callable may keep them.
        // Without an owner, nanobind would copy the chunk's vector, so gathered
        // valid rows go into an owned buffer instead, copied once.
        auto build_values = [&](idx_t i) {
            void* data = (void*)args[i].data;
            const TypeSpec* t = ctx->varargs_type ? ctx->varargs_type : ctx->param_types[i];
            nb::object owner;
            if (compact) {
                char* buf = owned_buffer(m * t->size, owner);
                for (idx_t j = 0; j < m; ++j) {
                    std::memcpy(buf + j * t->size, (char*)data + keep[j] * t->size, t->size);
                }
                data = buf;
            }
            nb::object values = nb::cast(
                nb::ndarray<nb::numpy, nb::ro>(data, 1, shape, owner, nullptr, t->dtype()));
            if (!ctx->special_handling) return values;
            return (nb::object)nb::make_tuple(values, build_input_mask(args[i].validity, n));
        };

        // Borrowed, since the holder owns it until thread teardown.
        void* init_data = nullptr;
        e.check(duckdb_v2_scalar_function_exec_get_init_data(info, &init_data, e.out()));
        auto* state_holder = (UDFState*)init_data;
        nb::handle state = state_holder ? nb::handle(state_holder->obj) : nb::handle();

        nb::object result;
        if (ctx->param_names.empty()) {
            nb::list args;
            if (state) args.append(state);
            for (idx_t i = 0; i < n_params; ++i) args.append(build_values(i));
            result = ctx->callable(*nb::tuple(args));
        } else {
            nb::dict kwargs;
            for (idx_t i = 0; i < n_params; ++i) {
                kwargs[ctx->param_names[i].c_str()] = build_values(i);
            }
            result = state ? ctx->callable(state, kwargs) : ctx->callable(kwargs);
        }

        // Returned value: an ndarray, or a (values, mask) tuple where
        // mask is a 1-D uint8/bool array (1=valid, 0=null).
        nb::object values_obj = result;
        nb::object mask_obj;
        if (nb::isinstance<nb::tuple>(result)) {
            nb::tuple t = nb::cast<nb::tuple>(result);
            if (nb::len(t) != 2) {
                throw DuckyError("ducky: UDF tuple return must be (values, mask)");
            }
            values_obj = t[0];
            mask_obj = t[1];
        }

        // Validate + memcpy the returned ndarray into the output vector.
        // nb::c_contig is load-bearing: the memcpy below reads `n` elements
        // straight off arr.data(), so a strided return (a slice such as
        // `big[::2]`) would otherwise be copied out wrong. nanobind
        // materializes a contiguous copy for us when needed.
        auto arr = nb::cast<nb::ndarray<nb::numpy, nb::c_contig>>(values_obj);
        if (arr.ndim() != 1 || arr.shape(0) != m) {
            throw DuckyError("ducky: UDF returned an array of shape != (" + std::to_string(m) +
                             ",)");
        }
        nb::dlpack::dtype want = ctx->return_type->dtype();
        nb::dlpack::dtype got = arr.dtype();
        if (got.code != want.code || got.bits != want.bits || got.lanes != want.lanes) {
            throw DuckyError("ducky: UDF returned an array of the wrong dtype for " +
                             std::string(ctx->return_type->name));
        }
        void* out_ptr = nullptr;
        e.check(duckdb_v2_vector_get_data_mutable(output, &out_ptr, e.out()));
        char* out_data = (char*)out_ptr;
        size_t width = ctx->return_type->size;
        if (compact) {
            for (idx_t j = 0; j < m; ++j) {
                std::memcpy(out_data + keep[j] * width, (const char*)arr.data() + j * width, width);
            }
        } else {
            std::memcpy(out_data, arr.data(), n * width);
        }

        if (mask_obj) {
            auto mask = nb::cast<nb::ndarray<nb::numpy, nb::c_contig>>(mask_obj);
            if (mask.ndim() != 1 || mask.shape(0) != m) {
                throw DuckyError("ducky: UDF mask shape != (" + std::to_string(m) + ",)");
            }
            nb::dlpack::dtype mdt = mask.dtype();
            // uint8 and bool are both 8-bit unsigned in DLPack (code=1, bits=8).
            if (mdt.bits != 8 || (mdt.code != (uint8_t)nb::dlpack::dtype_code::UInt &&
                                  mdt.code != (uint8_t)nb::dlpack::dtype_code::Bool)) {
                throw DuckyError("ducky: UDF mask must be uint8 or bool");
            }
            const uint8_t* valid = (const uint8_t*)mask.data();
            for (idx_t j = 0; j < m; ++j) {
                if (!valid[j]) set_row_null(output, out_validity, compact ? keep[j] : j);
            }
        }
    });
}

// ── Arrow UDF ────────────────────────────────────────────────────────────────

struct ArrowUDFContext {
    nb::handle callable;                                     // kept alive by the DuckDBHandle
    std::vector<std::string> param_names;                    // one entry per parameter
    std::vector<duckdb_v2_logical_type_handle> param_types;  // owned
    // Pre-computed views into param_names for zero-alloc trampoline calls.
    std::vector<const char*> param_name_ptrs;
    std::vector<idx_t> param_name_lengths;
    duckdb_v2_logical_type_handle return_type = nullptr;  // owned
    // Calling convention for the user's callable:
    //   record_batch=true  → fn(rb: pa.RecordBatch)
    //   record_batch=false, dict_style=false → fn(col0, col1, ...)  [positional pa.Array args]
    //   record_batch=false, dict_style=true  → fn({"name": col, ...})  [dict of pa.Array]
    bool record_batch = false;
    bool dict_style = false;
    // Arrow view of `return_type`, built lazily on the first call (registration
    // must not require pyarrow). Filled under the GIL by ensure_out_schema.
    nb::object out_schema;  // pa.Schema, one field
    nb::object out_type;    // that field's pa.DataType
    // Workers can enter ensure_out_schema concurrently on free-threaded builds.
    nb::ft_mutex out_schema_mu;

    ArrowUDFContext() = default;
    ArrowUDFContext(const ArrowUDFContext&) = delete;
    ArrowUDFContext& operator=(const ArrowUDFContext&) = delete;

    // Owns the types so a create_arrow_scalar_function that throws mid-parse
    // (e.g. an invalid second parameter type) frees the ones already built.
    ~ArrowUDFContext() {
        for (duckdb_v2_logical_type_handle& t : param_types) duckdb_v2_logical_type_destroy(&t);
        duckdb_v2_logical_type_destroy(&return_type);
    }
};

// RAII for an ArrowSchema/ArrowArray pair on the stack: releases whichever
// halves still own data when we leave scope (e.g. on exception).
struct ArrowPair {
    ArrowSchema schema{};
    ArrowArray array{};
    ~ArrowPair() {
        if (schema.release) schema.release(&schema);
        if (array.release) array.release(&array);
    }
};

// Owners of the v2 objects one Arrow UDF call converts through.
struct ExporterGuard {
    duckdb_v2_arrow_exporter_handle exporter = nullptr;
    ~ExporterGuard() { duckdb_v2_arrow_exporter_destroy(&exporter); }
};
struct ImporterGuard {
    duckdb_v2_arrow_importer_handle importer = nullptr;
    ~ImporterGuard() { duckdb_v2_arrow_importer_destroy(&importer); }
};
struct ChunkGuard {
    duckdb_v2_data_chunk_handle chunk = nullptr;
    ~ChunkGuard() { duckdb_v2_data_chunk_destroy(&chunk); }
};

// A worker's converters, reused for every chunk the worker processes. They
// borrow the query's client context, which outlives them.
struct ArrowUDFLocal {
    ExporterGuard in;   // the arguments, to Arrow
    ImporterGuard out;  // the result, from Arrow
};

void arrow_udf_local_destroy(void* ptr) { delete (ArrowUDFLocal*)ptr; }

void arrow_udf_init(duckdb_v2_scalar_function_init_info_handle info,
                    duckdb_v2_context_handle context, duckdb_v2_error_info_handle* err) {
    report(err, [&] {
        V2Error e;
        void* user_data = nullptr;
        e.check(duckdb_v2_scalar_function_init_get_user_data(info, &user_data, e.out()));
        auto* ctx = (ArrowUDFContext*)user_data;
        auto local = std::make_unique<ArrowUDFLocal>();
        e.check(duckdb_v2_arrow_exporter_create(
                    context, ctx->param_types.data(), ctx->param_name_ptrs.data(),
                    ctx->param_name_lengths.data(), ctx->param_types.size(), 0, &local->in.exporter,
                    e.out()),
                "ducky: failed to build the Arrow schema for UDF input: ");
        // The result's schema, as exporting the return type gives it.
        const std::string failed =
            "ducky: failed to build the Arrow schema for the UDF return type: ";
        const char* name = "x";
        idx_t name_length = 1;
        ExporterGuard result;
        e.check(duckdb_v2_arrow_exporter_create(context, &ctx->return_type, &name, &name_length, 1,
                                                0, &result.exporter, e.out()),
                failed);
        ArrowSchema schema{};
        e.check(duckdb_v2_arrow_exporter_get_schema(result.exporter, &schema, e.out()), failed);
        SchemaRelease schema_guard{&schema};
        e.check(duckdb_v2_arrow_importer_create(context, &schema, 0, &local->out.importer, e.out()),
                failed);
        duckdb_v2_opaque data{local.get(), &arrow_udf_local_destroy, nullptr};
        e.check(duckdb_v2_scalar_function_init_set_init_data(info, &data, e.out()));
        // NOLINTNEXTLINE(bugprone-unused-return-value)
        (void)local.release();
    });
}

// Cached on the context, since every chunk exports with it. Needs the GIL.
void ensure_out_schema(ArrowUDFContext* ctx, duckdb_v2_context_handle context) {
    nb::ft_lock_guard lock(ctx->out_schema_mu);
    if (ctx->out_type.is_valid()) return;
    const std::string failed = "ducky: failed to build the Arrow schema for the UDF return type: ";
    V2Error e;
    const char* name = "x";
    idx_t name_length = 1;
    ExporterGuard exporter;
    e.check(duckdb_v2_arrow_exporter_create(context, &ctx->return_type, &name, &name_length, 1, 0,
                                            &exporter.exporter, e.out()),
            failed);
    ArrowSchema target{};
    e.check(duckdb_v2_arrow_exporter_get_schema(exporter.exporter, &target, e.out()), failed);
    SchemaRelease guard{&target};
    nb::object schema = pyarrow_api().schema_import((uintptr_t)(&target));
    target.release = nullptr;  // pyarrow took ownership
    ctx->out_schema = schema;
    ctx->out_type = schema.attr("field")(0).attr("type");
}

void arrow_udf_trampoline(duckdb_v2_scalar_function_exec_info_handle info,
                          duckdb_v2_context_handle context, duckdb_v2_error_info_handle* err) {
    nb::gil_scoped_acquire gil;
    report_udf_error(err, "ducky UDF error: ", [&] {
        V2Error e;
        void* user_data = nullptr;
        e.check(duckdb_v2_scalar_function_exec_get_user_data(info, &user_data, e.out()));
        auto* ctx = (ArrowUDFContext*)user_data;
        void* init_data = nullptr;
        e.check(duckdb_v2_scalar_function_exec_get_init_data(info, &init_data, e.out()));
        auto* local = (ArrowUDFLocal*)init_data;
        idx_t n = 0;
        e.check(duckdb_v2_scalar_function_exec_get_row_count(info, &n, e.out()));
        idx_t n_cols = ctx->param_types.size();

        // The arguments as a chunk of their own, sharing their buffers.
        duckdb_v2_factory_handle factory = nullptr;
        e.check(duckdb_v2_context_get_factory(context, &factory, e.out()));
        ChunkGuard input;
        e.check(duckdb_v2_data_chunk_create(factory, ctx->param_types.data(), n_cols, &input.chunk,
                                            e.out()));
        for (idx_t i = 0; i < n_cols; ++i) {
            duckdb_v2_vector_handle arg = nullptr, column = nullptr;
            e.check(duckdb_v2_scalar_function_exec_get_arg(info, i, &arg, e.out()));
            e.check(duckdb_v2_data_chunk_get_vector(input.chunk, i, &column, e.out()));
            e.check(duckdb_v2_vector_reference(column, arg, e.out()));
        }

        const std::string failed_in = "ducky: failed to convert chunk to Arrow array: ";
        duckdb_v2_arrow_exporter_handle exporter = local->in.exporter;
        e.check(duckdb_v2_arrow_exporter_append(exporter, &input.chunk,
                                                /*consume=*/false, /*flush=*/true, e.out()),
                failed_in);
        ArrowPair in;
        e.check(duckdb_v2_arrow_exporter_get_schema(exporter, &in.schema, e.out()), failed_in);
        e.check(duckdb_v2_arrow_exporter_next_array(exporter, &in.array, e.out()), failed_in);
        if (!in.array.release) throw DuckyError(failed_in + "no array");

        // Hand the (array, schema) pair off to pyarrow via _import_from_c.
        // pyarrow takes ownership and clears the release callbacks on success.
        const PyArrowAPI& pa = pyarrow_api();
        nb::object rb = pa.import_from_c((uintptr_t)(&in.array), (uintptr_t)(&in.schema));

        // Dispatch to the user's callable according to the registered convention.
        nb::object result;
        if (ctx->record_batch) {
            result = ctx->callable(rb);
        } else if (ctx->dict_style) {
            nb::dict col_dict;
            for (idx_t i = 0; i < n_cols; ++i) {
                col_dict[ctx->param_names[i].c_str()] = rb.attr("column")(i);
            }
            result = ctx->callable(col_dict);
        } else {
            nb::list args;
            for (idx_t i = 0; i < n_cols; ++i) args.append(rb.attr("column")(i));
            result = ctx->callable(*nb::tuple(args));
        }

        // Wrap the user's pa.Array into a single-field RecordBatch, which
        // imports as a one-column chunk.
        //
        // The batch is pinned to the *declared* return type rather than to
        // whatever the callable happened to produce: pyarrow's kernels widen
        // types (pc.multiply on two DECIMAL(5,2) yields DECIMAL(11,4)), and a
        // widened array would reach duckdb_v2_vector_reference below with a
        // type the output vector doesn't share, which it rejects. Cast here
        // and let pyarrow reject a genuinely incompatible return.
        ensure_out_schema(ctx, context);
        if (!nb::cast<bool>(result.attr("type").attr("equals")(ctx->out_type))) {
            result = result.attr("cast")(ctx->out_type);
        }
        nb::list out_arrays;
        out_arrays.append(result);
        nb::object out_rb = pa.from_arrays(out_arrays, "schema"_a = ctx->out_schema);

        ArrowPair out;
        out_rb.attr("_export_to_c")((uintptr_t)(&out.array), (uintptr_t)(&out.schema));
        idx_t n_rows = (idx_t)out.array.length;
        if (n_rows != n) {
            throw DuckyError("ducky: arrow UDF returned " + std::to_string(n_rows) +
                             " rows; expected " + std::to_string(n));
        }

        const std::string failed_out = "ducky: failed to convert UDF output chunk: ";
        duckdb_v2_arrow_importer_handle importer = local->out.importer;
        // Consumed: the chunk references the batch's buffers, zero-copy.
        e.check(duckdb_v2_arrow_importer_append(importer, &out.array, /*consume=*/true,
                                                /*flush=*/true, e.out()),
                failed_out);
        ChunkGuard out_chunk;
        e.check(duckdb_v2_arrow_importer_next_chunk(importer, &out_chunk.chunk, e.out()),
                failed_out);
        if (!out_chunk.chunk) throw DuckyError(failed_out + "no chunk");
        duckdb_v2_vector_handle output = nullptr, column = nullptr;
        e.check(duckdb_v2_scalar_function_exec_get_result(info, &output, e.out()));
        e.check(duckdb_v2_data_chunk_get_vector(out_chunk.chunk, 0, &column, e.out()));
        e.check(duckdb_v2_vector_reference(output, column, e.out()));
    });
}

// Errors raise with `failed` before DuckDB's text.
class ScalarFunctionHandle {
   public:
    ScalarFunctionHandle(duckdb_v2_factory_handle factory, std::string failed)
        : failed_(std::move(failed)) {
        err_.check(duckdb_v2_scalar_function_create(factory, &f_, err_.out()), failed_);
    }
    ~ScalarFunctionHandle() {
        // Destroying an unregistered function drops the last reference to its
        // user data, whose destructor takes the GIL.
        release_gil_if_held release;
        duckdb_v2_scalar_function_destroy(&f_);
    }
    ScalarFunctionHandle(const ScalarFunctionHandle&) = delete;
    ScalarFunctionHandle& operator=(const ScalarFunctionHandle&) = delete;

    duckdb_v2_scalar_function_handle get() const { return f_; }

    // Hands `data` to the function, which destroys it under the GIL.
    template <typename T>
    void set_user_data(std::unique_ptr<T> data) {
        duckdb_v2_opaque opaque{data.get(), &gil_delete<T>, nullptr};
        err_.check(duckdb_v2_scalar_function_set_user_data(f_, &opaque, err_.out()), failed_);
        // NOLINTNEXTLINE(bugprone-unused-return-value)
        (void)data.release();
    }

    // Registration takes the client context lock, which a query running Python
    // callbacks on another thread may hold while it waits for the GIL.
    void register_on(duckdb_v2_connection_handle connection) {
        DUCKDB_V2_ERROR code;
        {
            release_gil_if_held release;
            code = duckdb_v2_connection_register_scalar_function(connection, f_, err_.out());
        }
        err_.check(code, failed_);
    }

   private:
    duckdb_v2_scalar_function_handle f_ = nullptr;
    V2Error err_;
    std::string failed_;
};

}  // namespace

const TypeSpec& lookup_typespec(const std::string& name) { return lookup(name); }

const StructDtypes& struct_dtypes() {
    static std::atomic<StructDtypes*> cached{nullptr};
    static nb::ft_mutex mu;
    return cached_singleton(cached, mu, [] {
        nb::module_ np = nb::module_::import_("numpy");
        auto mk = [&](std::initializer_list<std::pair<const char*, const char*>> fields) {
            nb::list spec;
            for (auto& [name, code] : fields) spec.append(nb::make_tuple(name, code));
            return np.attr("dtype")(spec);
        };
        return StructDtypes{
            mk({{"lower", "<u8"}, {"upper", "<i8"}}),
            mk({{"lower", "<u8"}, {"upper", "<u8"}}),
            mk({{"months", "<i4"}, {"days", "<i4"}, {"micros", "<i8"}}),
        };
    });
}

const TypeSpec* typespec_for(duckdb_type type) {
    for (const TypeSpec& t : kTypes) {
        if (t.type == type) return &t;
    }
    return nullptr;
}

bool array_layout(duckdb_logical_type logical, ArrayLayout& out) {
    if (duckdb_get_type_id(logical) != DUCKDB_TYPE_ARRAY) return false;
    out = ArrayLayout();
    duckdb_logical_type current = logical;
    bool owned = false;  // `logical` belongs to the caller
    while (duckdb_get_type_id(current) == DUCKDB_TYPE_ARRAY) {
        size_t size = (size_t)duckdb_array_type_array_size(current);
        out.dims.push_back(size);
        out.values_per_row *= size;
        duckdb_logical_type child = duckdb_array_type_child_type(current);
        if (owned) duckdb_destroy_logical_type(&current);
        current = child;
        owned = true;
    }
    out.leaf = typespec_for(duckdb_get_type_id(current));
    duckdb_destroy_logical_type(&current);
    return out.leaf != nullptr;
}

nb::list enum_categories(duckdb_logical_type logical, const std::string& name) {
    duckdb_type tid = duckdb_get_type_id(logical);
    if (tid != DUCKDB_TYPE_ENUM) {
        throw DuckyError("ducky: categories: column '" + name + "' is " + duckdb_type_name(tid) +
                         ", not ENUM");
    }
    nb::list out;
    uint32_t n = duckdb_enum_dictionary_size(logical);
    for (uint32_t i = 0; i < n; ++i) {
        char* value = duckdb_enum_dictionary_value(logical, i);
        nb::str label(value);
        duckdb_free(value);
        out.append(label);
    }
    return out;
}

void create_scalar_function(Connection& con, const std::string& name, nb::callable fn,
                            nb::object parameters, nb::object return_type, nb::object varargs,
                            nb::object init, bool is_volatile, bool special_handling) {
    std::shared_ptr<DuckDBHandle> handle = con.open_handle();
    auto ctx = std::make_unique<UDFContext>();
    bool has_varargs = !varargs.is_none();
    if (has_varargs && !parameters.is_none()) {
        throw DuckyError("ducky: UDF '" + name + "' cannot set both `parameters` and `varargs`");
    }
    if (!has_varargs && parameters.is_none()) parameters = infer_parameters(fn);
    std::string rt_name =
        return_type.is_none() ? infer_return_type(fn) : nb::cast<std::string>(return_type);
    ctx->callable = handle->keep(std::move(fn));
    ctx->return_type = &lookup(rt_name);
    ctx->special_handling = special_handling;
    if (!init.is_none()) {
        try {
            ctx->init_callable = handle->keep(nb::cast<nb::callable>(init));
        } catch (const nb::cast_error&) {
            throw DuckyError("ducky: UDF '" + name + "' init= must be a callable");
        }
    }

    if (has_varargs) {
        ctx->varargs_type = &lookup(nb::cast<std::string>(varargs));
    } else if (nb::isinstance<nb::dict>(parameters)) {
        nb::dict d = nb::cast<nb::dict>(parameters);
        for (auto item : d) {
            ctx->param_names.emplace_back(nb::cast<std::string>(item.first));
            ctx->param_types.push_back(&lookup(nb::cast<std::string>(item.second)));
        }
    } else {
        nb::sequence seq = nb::cast<nb::sequence>(parameters);
        for (auto item : seq) {
            ctx->param_types.push_back(&lookup(nb::cast<std::string>(item)));
        }
    }
    if (!has_varargs && ctx->param_types.empty()) {
        throw DuckyError("ducky: UDF '" + name + "' must declare at least one parameter");
    }

    const std::string failed = "ducky: failed to register UDF '" + name + "': ";
    V2Error err;
    auto check = [&](DUCKDB_V2_ERROR code) { err.check(code, failed); };
    duckdb_v2_factory_handle factory = nullptr;
    check(duckdb_v2_connection_get_factory(handle->connection_v2, &factory, err.out()));
    // Adds a parameter or sets the return type to a type id, through `add`.
    auto with_type = [&](DUCKDB_V2_LOGICAL_TYPE_ID id, auto add) {
        duckdb_v2_logical_type_handle t = nullptr;
        check(duckdb_v2_factory_create_type_from_id(factory, id, nullptr, nullptr, nullptr, 0, &t,
                                                    err.out()));
        DUCKDB_V2_ERROR code = add(t);
        duckdb_v2_logical_type_destroy(&t);
        check(code);
    };

    ScalarFunctionHandle f(factory, failed);
    check(duckdb_v2_scalar_function_set_name(f.get(), name.data(), name.size(), err.out()));
    duckdb_v2_function_signature_handle sig = nullptr;
    check(duckdb_v2_scalar_function_get_signature(f.get(), &sig, err.out()));
    if (has_varargs) {
        with_type(ctx->varargs_type->v2_type, [&](duckdb_v2_logical_type_handle t) {
            return duckdb_v2_function_signature_add_parameter(
                sig, "args", 4, t, nullptr, DUCKDB_V2_FUNCTION_PARAMETER_KIND_POSITIONAL_VARIADIC,
                err.out());
        });
    }
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
    check(duckdb_v2_scalar_function_set_exec_callback(f.get(), &udf_trampoline, err.out()));
    if (is_volatile) {
        check(duckdb_v2_scalar_function_set_property(f.get(), DUCKDB_V2_FUNCTION_PROPERTY_STABILITY,
                                                     DUCKDB_V2_FUNCTION_PROPERTY_STABILITY_VOLATILE,
                                                     err.out()));
    }
    if (special_handling) {
        check(duckdb_v2_scalar_function_set_property(
            f.get(), DUCKDB_V2_FUNCTION_PROPERTY_NULL_HANDLING,
            DUCKDB_V2_FUNCTION_PROPERTY_NULL_HANDLING_SPECIAL, err.out()));
    }
    if (ctx->init_callable) {
        check(
            duckdb_v2_scalar_function_set_init_callback(f.get(), &udf_init_trampoline, err.out()));
    }
    f.set_user_data(std::move(ctx));
    f.register_on(handle->connection_v2);
}

void create_arrow_scalar_function(Connection& con, const std::string& name, nb::callable fn,
                                  nb::object parameters, const std::string& return_type,
                                  bool record_batch) {
    std::shared_ptr<DuckDBHandle> handle = con.open_handle();
    const std::string failed = "ducky: failed to register arrow UDF '" + name + "': ";
    V2Error err;
    duckdb_v2_factory_handle types = nullptr;
    err.check(duckdb_v2_connection_get_factory(handle->connection_v2, &types, err.out()), failed);
    auto ctx = std::make_unique<ArrowUDFContext>();
    ctx->callable = handle->keep(std::move(fn));
    ctx->record_batch = record_batch;

    if (nb::isinstance<nb::dict>(parameters)) {
        ctx->dict_style = true;
        nb::dict d = nb::cast<nb::dict>(parameters);
        for (auto item : d) {
            ctx->param_names.emplace_back(nb::cast<std::string>(item.first));
            ctx->param_types.push_back(v2_parse_type(types, nb::cast<std::string>(item.second)));
        }
    } else {
        nb::sequence seq = nb::cast<nb::sequence>(parameters);
        idx_t i = 0;
        for (auto item : seq) {
            ctx->param_names.emplace_back("arg" + std::to_string(i++));
            ctx->param_types.push_back(v2_parse_type(types, nb::cast<std::string>(item)));
        }
    }
    if (ctx->param_types.empty()) {
        throw DuckyError("ducky: arrow UDF '" + name + "' must declare at least one parameter");
    }
    ctx->return_type = v2_parse_type(types, return_type);

    // Pre-compute stable pointers into param_names for zero-alloc trampoline
    // calls. param_names is never modified after this point.
    for (const std::string& n : ctx->param_names) {
        ctx->param_name_ptrs.push_back(n.data());
        ctx->param_name_lengths.push_back(n.size());
    }

    auto check = [&](DUCKDB_V2_ERROR code) { err.check(code, failed); };
    ScalarFunctionHandle f(types, failed);
    check(duckdb_v2_scalar_function_set_name(f.get(), name.data(), name.size(), err.out()));
    duckdb_v2_function_signature_handle sig = nullptr;
    check(duckdb_v2_scalar_function_get_signature(f.get(), &sig, err.out()));
    for (size_t i = 0; i < ctx->param_types.size(); ++i) {
        const std::string& param = ctx->param_names[i];
        check(duckdb_v2_function_signature_add_parameter(
            sig, param.data(), param.size(), ctx->param_types[i], nullptr,
            DUCKDB_V2_FUNCTION_PARAMETER_KIND_POSITIONAL_ONLY, err.out()));
    }
    check(duckdb_v2_function_signature_set_return_type(sig, ctx->return_type, err.out()));
    check(duckdb_v2_scalar_function_set_init_callback(f.get(), &arrow_udf_init, err.out()));
    check(duckdb_v2_scalar_function_set_exec_callback(f.get(), &arrow_udf_trampoline, err.out()));
    f.set_user_data(std::move(ctx));
    f.register_on(handle->connection_v2);
}
