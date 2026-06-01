#include "function.hpp"

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <cstring>
#include <utility>

#include "connection.hpp"
#include "ducky.hpp"

namespace {

struct TypeSpec {
    const char* name;
    duckdb_type type;
    size_t size;  // size in bytes of one element in the chunk buffer
    nb::dlpack::dtype (*dtype)();
};

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

// Lives in DuckDB's extra_info slot; carries everything the trampoline needs to
// dispatch one chunk of input to the user's callable.
struct UDFContext {
    nb::callable callable;  // owns the Python ref
    // If non-empty: dict-style — callable receives one dict {name: ndarray}.
    // If empty: positional — callable receives ndarrays as positional args.
    std::vector<std::string> param_names;
    std::vector<const TypeSpec*> param_types;
    const TypeSpec* return_type;
};

void udf_extra_info_destroy(void* ptr) {
    if (!ptr) return;
    nb::gil_scoped_acquire gil;
    delete static_cast<UDFContext*>(ptr);
}

void udf_trampoline(duckdb_function_info info, duckdb_data_chunk input, duckdb_vector output) {
    auto* ctx = static_cast<UDFContext*>(duckdb_scalar_function_get_extra_info(info));
    nb::gil_scoped_acquire gil;

    idx_t n = duckdb_data_chunk_get_size(input);
    idx_t n_params = ctx->param_types.size();
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
                args.append(nb::ndarray<nb::numpy, nb::ro>(data, 1, shape, nb::handle(), nullptr,
                                                           ctx->param_types[i]->dtype()));
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
        std::string msg = std::string("ducky UDF error: ") + e.what();
        duckdb_scalar_function_set_error(info, msg.c_str());
    } catch (const std::exception& e) {
        std::string msg = std::string("ducky UDF error: ") + e.what();
        duckdb_scalar_function_set_error(info, msg.c_str());
    }
}

}  // namespace

duckdb_type parse_type_name(const std::string& name) { return lookup(name).type; }

void create_scalar_function(Connection& con, const std::string& name, nb::callable fn,
                            nb::object parameters, nb::object return_type) {
    auto ctx = std::make_unique<UDFContext>();
    ctx->callable = std::move(fn);
    ctx->return_type = &lookup(return_type);

    if (nb::isinstance<nb::dict>(parameters)) {
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
    if (ctx->param_types.empty()) {
        throw DuckyError("ducky: UDF '" + name + "' must declare at least one parameter");
    }

    duckdb_scalar_function f = duckdb_create_scalar_function();
    duckdb_scalar_function_set_name(f, name.c_str());
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
    (void)ctx.release();
}
