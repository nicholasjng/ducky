#include "function.hpp"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstring>
#include <utility>
#include <vector>

#include "arrow_abi.h"
#include "connection.hpp"
#include "ducky.hpp"

namespace {

template <typename T>
nb::dlpack::dtype dt() {
    return nb::dtype<T>();
}

// The set of primitive types we support for UDF parameters and return values.
// Mirrors the set Chunk.column() exposes — see src/cpp/chunk.cpp.
const TypeSpec kTypes[] = {
    {"BOOLEAN", DUCKDB_TYPE_BOOLEAN, sizeof(bool), &dt<bool>},
    {"TINYINT", DUCKDB_TYPE_TINYINT, sizeof(int8_t), &dt<int8_t>},
    {"SMALLINT", DUCKDB_TYPE_SMALLINT, sizeof(int16_t), &dt<int16_t>},
    {"INTEGER", DUCKDB_TYPE_INTEGER, sizeof(int32_t), &dt<int32_t>},
    {"BIGINT", DUCKDB_TYPE_BIGINT, sizeof(int64_t), &dt<int64_t>},
    {"UTINYINT", DUCKDB_TYPE_UTINYINT, sizeof(uint8_t), &dt<uint8_t>},
    {"USMALLINT", DUCKDB_TYPE_USMALLINT, sizeof(uint16_t), &dt<uint16_t>},
    {"UINTEGER", DUCKDB_TYPE_UINTEGER, sizeof(uint32_t), &dt<uint32_t>},
    {"UBIGINT", DUCKDB_TYPE_UBIGINT, sizeof(uint64_t), &dt<uint64_t>},
    {"FLOAT", DUCKDB_TYPE_FLOAT, sizeof(float), &dt<float>},
    {"DOUBLE", DUCKDB_TYPE_DOUBLE, sizeof(double), &dt<double>},
    {"DATE", DUCKDB_TYPE_DATE, sizeof(int32_t), &dt<int32_t>},
    {"TIME", DUCKDB_TYPE_TIME, sizeof(int64_t), &dt<int64_t>},
    {"TIMESTAMP", DUCKDB_TYPE_TIMESTAMP, sizeof(int64_t), &dt<int64_t>},
    {"TIMESTAMP_S", DUCKDB_TYPE_TIMESTAMP_S, sizeof(int64_t), &dt<int64_t>},
    {"TIMESTAMP_MS", DUCKDB_TYPE_TIMESTAMP_MS, sizeof(int64_t), &dt<int64_t>},
    {"TIMESTAMP_NS", DUCKDB_TYPE_TIMESTAMP_NS, sizeof(int64_t), &dt<int64_t>},
    {"TIMESTAMP_TZ", DUCKDB_TYPE_TIMESTAMP_TZ, sizeof(int64_t), &dt<int64_t>},
};

const TypeSpec& lookup(const std::string& name) {
    for (const TypeSpec& t : kTypes) {
        if (name == t.name) return t;
    }
    throw DuckyError("ducky: unknown UDF type '" + name +
                     "' (supported: BOOLEAN, [U]TINYINT..BIGINT, FLOAT, DOUBLE, "
                     "DATE, TIME, TIMESTAMP[_S/MS/NS/TZ])");
}

// Map a Python annotation object to a DuckDB type string. Only the natural
// numeric/boolean Python types are recognized — anything else is a user
// error that should be made explicit by passing `parameters=` / `return_type=`.
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

// Build a positional `parameters` list by inspecting `fn`'s annotations.
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

// Lives in DuckDB's extra_info slot; carries everything the trampoline needs to
// dispatch one chunk of input to the user's callable.
struct UDFContext {
    nb::callable callable;  // owns the Python ref
    // If non-empty: dict-style — callable receives one dict {name: ndarray}.
    // If empty: positional — callable receives ndarrays as positional args.
    std::vector<std::string> param_names;
    std::vector<const TypeSpec*> param_types;
    const TypeSpec* return_type;
    // When set, the function takes a variable number of arguments, all of this
    // type. `param_types` is ignored; arity is read from the input chunk.
    const TypeSpec* varargs_type = nullptr;
};

void udf_extra_info_destroy(void* ptr) {
    if (!ptr) return;
    nb::gil_scoped_acquire gil;
    delete static_cast<UDFContext*>(ptr);
}

// Forwards a C++ or Python exception as a DuckDB UDF error string.
void set_udf_error(duckdb_function_info info, const char* what) {
    std::string msg = std::string("ducky UDF error: ") + what;
    duckdb_scalar_function_set_error(info, msg.c_str());
}

void udf_trampoline(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    auto* ctx = static_cast<UDFContext*>(duckdb_scalar_function_get_extra_info(info));
    nb::gil_scoped_acquire gil;

    idx_t n = duckdb_data_chunk_get_size(input);
    idx_t n_params =
        ctx->varargs_type ? duckdb_data_chunk_get_column_count(input) : ctx->param_types.size();
    size_t shape[1] = {(size_t)n};

    try {
        // Build input ndarray views over the chunk's vectors. Owner is empty:
        // the arrays are valid only for the duration of this call. Capturing
        // them past the call is undefined — same hazard as Chunk.column views.
        nb::object result;
        if (ctx->param_names.empty()) {
            nb::list args;
            for (idx_t i = 0; i < n_params; ++i) {
                duckdb_vector v = duckdb_data_chunk_get_vector(input, i);
                void* data = duckdb_vector_get_data(v);
                const TypeSpec* t = ctx->varargs_type ? ctx->varargs_type : ctx->param_types[i];
                args.append(nb::ndarray<nb::numpy, nb::ro>(data, 1, shape, nb::handle(), nullptr,
                                                           t->dtype()));
            }
            result = ctx->callable(*nb::tuple(args));
        } else {
            nb::dict kwargs;
            for (idx_t i = 0; i < n_params; ++i) {
                duckdb_vector v = duckdb_data_chunk_get_vector(input, i);
                void* data = duckdb_vector_get_data(v);
                kwargs[ctx->param_names[i].c_str()] = nb::ndarray<nb::numpy, nb::ro>(
                    data, 1, shape, nb::handle(), nullptr, ctx->param_types[i]->dtype());
            }
            result = ctx->callable(kwargs);
        }

        // The UDF may return either an ndarray of values, or a (values, mask)
        // tuple where `mask` is a 1D uint8/bool array (1=valid, 0=null).
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
        auto arr = nb::cast<nb::ndarray<nb::numpy>>(values_obj);
        if (arr.ndim() != 1 || arr.shape(0) != n) {
            throw DuckyError("ducky: UDF returned an array of shape != (" + std::to_string(n) +
                             ",)");
        }
        nb::dlpack::dtype want = ctx->return_type->dtype();
        nb::dlpack::dtype got = arr.dtype();
        if (got.code != want.code || got.bits != want.bits || got.lanes != want.lanes) {
            throw DuckyError("ducky: UDF returned an array of the wrong dtype for " +
                             std::string(ctx->return_type->name));
        }
        void* out_data = duckdb_vector_get_data(output);
        std::memcpy(out_data, arr.data(), n * ctx->return_type->size);

        if (mask_obj) {
            auto mask = nb::cast<nb::ndarray<nb::numpy>>(mask_obj);
            if (mask.ndim() != 1 || mask.shape(0) != n) {
                throw DuckyError("ducky: UDF mask shape != (" + std::to_string(n) + ",)");
            }
            nb::dlpack::dtype mdt = mask.dtype();
            // Accept uint8 or bool (both 8-bit unsigned in DLPack: code=1, bits=8).
            if (mdt.bits != 8 || (mdt.code != (uint8_t)nb::dlpack::dtype_code::UInt &&
                                  mdt.code != (uint8_t)nb::dlpack::dtype_code::Bool)) {
                throw DuckyError("ducky: UDF mask must be uint8 or bool");
            }
            duckdb_vector_ensure_validity_writable(output);
            uint64_t* validity = duckdb_vector_get_validity(output);
            const uint8_t* m = (const uint8_t*)mask.data();
            for (idx_t i = 0; i < n; ++i) {
                if (!m[i]) duckdb_validity_set_row_invalid(validity, i);
            }
        }
    } catch (nb::python_error& e) {
        set_udf_error(info, e.what());
    } catch (const std::exception& e) {
        set_udf_error(info, e.what());
    }
}

// ── Arrow UDF ────────────────────────────────────────────────────────────────

struct ArrowUDFContext {
    nb::callable callable;            // owns the Python ref
    duckdb_connection con = nullptr;  // borrowed; valid while connection is open
    duckdb_arrow_options options = nullptr;
    std::vector<std::string> param_names;          // one entry per parameter
    std::vector<duckdb_logical_type> param_types;  // owned
    // Pre-computed views into param_names / param_types for zero-alloc trampoline calls.
    std::vector<const char*> param_name_ptrs;
    duckdb_logical_type return_type = nullptr;  // owned
    // Calling convention for the user's callable:
    //   record_batch=true  → fn(rb: pa.RecordBatch)
    //   record_batch=false, dict_style=false → fn(col0, col1, ...)  [positional pa.Array args]
    //   record_batch=false, dict_style=true  → fn({"name": col, ...})  [dict of pa.Array]
    bool record_batch = false;
    bool dict_style = false;
};

void arrow_udf_extra_info_destroy(void* ptr) {
    if (!ptr) return;
    auto* ctx = static_cast<ArrowUDFContext*>(ptr);
    for (duckdb_logical_type& t : ctx->param_types) duckdb_destroy_logical_type(&t);
    if (ctx->return_type) duckdb_destroy_logical_type(&ctx->return_type);
    if (ctx->options) duckdb_destroy_arrow_options(&ctx->options);
    {
        nb::gil_scoped_acquire gil;
        delete ctx;
    }
}

// Drains a duckdb_error_data into a std::string. Returns true if an error was
// present (and copied out); the error_data is always destroyed.
bool drain_arrow_error(duckdb_error_data err, std::string& out) {
    if (!err) return false;
    bool has = duckdb_error_data_has_error(err);
    if (has) out = duckdb_error_data_message(err);
    duckdb_destroy_error_data(&err);
    return has;
}

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

// RAII wrappers for DuckDB heap objects used in the Arrow trampoline.
struct ChunkGuard {
    duckdb_data_chunk chunk = nullptr;
    explicit ChunkGuard(duckdb_data_chunk c) : chunk(c) {}
    ~ChunkGuard() {
        if (chunk) duckdb_destroy_data_chunk(&chunk);
    }
};
struct ConvertedSchemaGuard {
    duckdb_arrow_converted_schema schema = nullptr;
    explicit ConvertedSchemaGuard(duckdb_arrow_converted_schema s) : schema(s) {}
    ~ConvertedSchemaGuard() {
        if (schema) duckdb_destroy_arrow_converted_schema(&schema);
    }
};

void arrow_udf_trampoline(duckdb_function_info info, duckdb_data_chunk input,
                          duckdb_vector output) {
    auto* ctx = static_cast<ArrowUDFContext*>(duckdb_scalar_function_get_extra_info(info));
    nb::gil_scoped_acquire gil;
    idx_t n_cols = duckdb_data_chunk_get_column_count(input);

    try {
        ArrowPair in;
        std::string err;
        if (drain_arrow_error(
                duckdb_to_arrow_schema(ctx->options, ctx->param_types.data(),
                                       ctx->param_name_ptrs.data(), n_cols, &in.schema),
                err)) {
            throw DuckyError("ducky: failed to build Arrow schema for UDF input: " + err);
        }
        if (drain_arrow_error(duckdb_data_chunk_to_arrow(ctx->options, input, &in.array), err)) {
            throw DuckyError("ducky: failed to convert chunk to Arrow array: " + err);
        }

        // Hand the (array, schema) pair off to pyarrow via _import_from_c.
        // pyarrow takes ownership and clears the release callbacks on success.
        nb::module_ pa = nb::module_::import_("pyarrow");
        nb::object rb =
            pa.attr("RecordBatch")
                .attr("_import_from_c")((uintptr_t)(&in.array), (uintptr_t)(&in.schema));

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

        // Wrap the user's pa.Array into a single-field RecordBatch so we can
        // round-trip back through duckdb_data_chunk_from_arrow (which expects a
        // struct-shaped ArrowArray representing one chunk).
        nb::list out_arrays, out_names;
        out_arrays.append(result);
        out_names.append(nb::str("x"));
        nb::object out_rb = pa.attr("RecordBatch").attr("from_arrays")(out_arrays, out_names);

        ArrowPair out;
        out_rb.attr("_export_to_c")((uintptr_t)(&out.array), (uintptr_t)(&out.schema));

        duckdb_arrow_converted_schema raw_converted = nullptr;
        if (drain_arrow_error(duckdb_schema_from_arrow(ctx->con, &out.schema, &raw_converted),
                              err)) {
            if (raw_converted) duckdb_destroy_arrow_converted_schema(&raw_converted);
            throw DuckyError("ducky: failed to convert UDF output schema: " + err);
        }
        ConvertedSchemaGuard converted(raw_converted);

        duckdb_data_chunk raw_chunk = nullptr;
        if (drain_arrow_error(
                duckdb_data_chunk_from_arrow(ctx->con, &out.array, converted.schema, &raw_chunk),
                err)) {
            if (raw_chunk) duckdb_destroy_data_chunk(&raw_chunk);
            throw DuckyError("ducky: failed to convert UDF output chunk: " + err);
        }
        // duckdb_data_chunk_from_arrow has taken ownership of out.array.
        out.array.release = nullptr;
        ChunkGuard out_chunk(raw_chunk);

        idx_t n_rows = duckdb_data_chunk_get_size(out_chunk.chunk);
        idx_t expected = duckdb_data_chunk_get_size(input);
        if (n_rows != expected) {
            throw DuckyError("ducky: arrow UDF returned " + std::to_string(n_rows) +
                             " rows; expected " + std::to_string(expected));
        }
        duckdb_vector v = duckdb_data_chunk_get_vector(out_chunk.chunk, 0);
        duckdb_vector_reference_vector(output, v);
    } catch (nb::python_error& e) {
        set_udf_error(info, e.what());
    } catch (const std::exception& e) {
        set_udf_error(info, e.what());
    }
}

}  // namespace

const TypeSpec& lookup_typespec(const std::string& name) { return lookup(name); }

duckdb_logical_type parse_logical_type(duckdb_connection con, const std::string& type_str) {
    if (type_str.empty()) {
        throw DuckyError("ducky: empty type string");
    }
    for (char c : type_str) {
        bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                  c == '_' || c == '(' || c == ')' || c == '[' || c == ']' || c == '<' ||
                  c == '>' || c == ',' || c == ' ';
        if (!ok) {
            throw DuckyError("ducky: invalid character in type string '" + type_str + "'");
        }
    }
    std::string q = "SELECT CAST(NULL AS " + type_str + ")";
    duckdb_result r;
    if (duckdb_query(con, q.c_str(), &r) == DuckDBError) {
        std::string err = duckdb_result_error(&r);
        duckdb_destroy_result(&r);
        throw DuckyError("ducky: invalid DuckDB type '" + type_str + "': " + err);
    }
    duckdb_logical_type lt = duckdb_column_logical_type(&r, 0);
    duckdb_destroy_result(&r);
    return lt;
}

const TypeSpec* typespec_for(duckdb_type type) {
    for (const TypeSpec& t : kTypes) {
        if (t.type == type) return &t;
    }
    return nullptr;
}

void create_scalar_function(Connection& con, const std::string& name, nb::callable fn,
                            nb::object parameters, nb::object return_type, nb::object varargs) {
    auto ctx = std::make_unique<UDFContext>();
    bool has_varargs = !varargs.is_none();
    if (has_varargs && !parameters.is_none()) {
        throw DuckyError("ducky: UDF '" + name + "' cannot set both `parameters` and `varargs`");
    }
    if (!has_varargs && parameters.is_none()) parameters = infer_parameters(fn);
    std::string rt_name =
        return_type.is_none() ? infer_return_type(fn) : nb::cast<std::string>(return_type);
    ctx->callable = std::move(fn);
    ctx->return_type = &lookup(rt_name);

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

    duckdb_scalar_function f = duckdb_create_scalar_function();
    duckdb_scalar_function_set_name(f, name.c_str());
    if (has_varargs) {
        duckdb_logical_type lt = duckdb_create_logical_type(ctx->varargs_type->type);
        duckdb_scalar_function_set_varargs(f, lt);
        duckdb_destroy_logical_type(&lt);
    }
    for (const TypeSpec* t : ctx->param_types) {
        duckdb_logical_type lt = duckdb_create_logical_type(t->type);
        duckdb_scalar_function_add_parameter(f, lt);
        duckdb_destroy_logical_type(&lt);
    }
    duckdb_logical_type rt = duckdb_create_logical_type(ctx->return_type->type);
    duckdb_scalar_function_set_return_type(f, rt);
    duckdb_destroy_logical_type(&rt);
    duckdb_scalar_function_set_function(f, &udf_trampoline);
    // DuckDB takes ownership of ctx via the destroy callback; release on success.
    duckdb_scalar_function_set_extra_info(f, ctx.get(), &udf_extra_info_destroy);

    duckdb_connection raw = con.raw_connection();
    duckdb_state state = duckdb_register_scalar_function(raw, f);
    duckdb_destroy_scalar_function(&f);
    if (state == DuckDBError) {
        // Registration failed: DuckDB didn't take ownership of extra_info, so
        // ctx is still ours and unique_ptr will free it.
        throw DuckyError("ducky: failed to register UDF '" + name +
                         "' (name in use, or invalid types?)");
    }
    // Hand off ownership of ctx to DuckDB; release() returns the raw pointer
    // (already stashed via set_extra_info above), which we intentionally drop.
    // NOLINTNEXTLINE(bugprone-unused-return-value)
    (void)ctx.release();
}

void create_arrow_scalar_function(Connection& con, const std::string& name, nb::callable fn,
                                  nb::object parameters, const std::string& return_type,
                                  bool record_batch) {
    duckdb_connection raw = con.raw_connection();
    auto ctx = std::make_unique<ArrowUDFContext>();
    ctx->callable = std::move(fn);
    ctx->con = raw;
    ctx->record_batch = record_batch;
    duckdb_connection_get_arrow_options(raw, &ctx->options);
    if (!ctx->options) {
        throw DuckyError("ducky: failed to fetch Arrow options from connection");
    }

    if (nb::isinstance<nb::dict>(parameters)) {
        ctx->dict_style = true;
        nb::dict d = nb::cast<nb::dict>(parameters);
        for (auto item : d) {
            ctx->param_names.emplace_back(nb::cast<std::string>(item.first));
            ctx->param_types.push_back(parse_logical_type(raw, nb::cast<std::string>(item.second)));
        }
    } else {
        nb::sequence seq = nb::cast<nb::sequence>(parameters);
        idx_t i = 0;
        for (auto item : seq) {
            ctx->param_names.emplace_back("arg" + std::to_string(i++));
            ctx->param_types.push_back(parse_logical_type(raw, nb::cast<std::string>(item)));
        }
    }
    if (ctx->param_types.empty()) {
        throw DuckyError("ducky: arrow UDF '" + name + "' must declare at least one parameter");
    }
    ctx->return_type = parse_logical_type(raw, return_type);

    // Pre-compute stable const char* pointers into param_names for zero-alloc
    // trampoline calls. param_names is never modified after this point.
    ctx->param_name_ptrs.reserve(ctx->param_names.size());
    for (const std::string& n : ctx->param_names) ctx->param_name_ptrs.push_back(n.c_str());

    duckdb_scalar_function f = duckdb_create_scalar_function();
    duckdb_scalar_function_set_name(f, name.c_str());
    for (duckdb_logical_type t : ctx->param_types) {
        duckdb_scalar_function_add_parameter(f, t);
    }
    duckdb_scalar_function_set_return_type(f, ctx->return_type);
    duckdb_scalar_function_set_function(f, &arrow_udf_trampoline);
    duckdb_scalar_function_set_extra_info(f, ctx.get(), &arrow_udf_extra_info_destroy);

    duckdb_state state = duckdb_register_scalar_function(raw, f);
    duckdb_destroy_scalar_function(&f);
    if (state == DuckDBError) {
        throw DuckyError("ducky: failed to register arrow UDF '" + name +
                         "' (name in use, or invalid types?)");
    }
    // NOLINTNEXTLINE(bugprone-unused-return-value)
    (void)ctx.release();
}
