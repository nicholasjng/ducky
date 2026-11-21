#include "table.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <cstring>
#include <memory>

#include "connection.hpp"
#include "ducky.hpp"
#include "py_datetime.hpp"

namespace nb = nanobind;

namespace {

// `factory` renders types without a Python counterpart as text.
nb::object value_to_python(duckdb_v2_factory_handle factory, duckdb_v2_value_handle val) {
    V2Error e;
    bool is_null = false;
    e.check(duckdb_v2_value_is_null(val, &is_null, e.out()));
    if (is_null) return nb::none();
    duckdb_v2_logical_type_handle type = nullptr;
    e.check(duckdb_v2_value_get_logical_type(val, &type, e.out()));
    DUCKDB_V2_LOGICAL_TYPE_ID tid = DUCKDB_V2_LOGICAL_TYPE_ID_INVALID;
    DUCKDB_V2_ERROR code = duckdb_v2_logical_type_get_id(type, &tid, e.out());
    duckdb_v2_logical_type_destroy(&type);
    e.check(code);

    // Reads the payload through `getter`, which converts it to its own type.
    auto get = [&](auto getter, auto out) {
        e.check(getter(val, &out, e.out()));
        return out;
    };
    switch (tid) {
        case DUCKDB_V2_LOGICAL_TYPE_ID_BOOLEAN:
            return nb::bool_(get(duckdb_v2_value_get_bool, false));
        case DUCKDB_V2_LOGICAL_TYPE_ID_TINYINT:
            return nb::int_(get(duckdb_v2_value_get_tinyint, (int8_t)0));
        case DUCKDB_V2_LOGICAL_TYPE_ID_SMALLINT:
            return nb::int_(get(duckdb_v2_value_get_smallint, (int16_t)0));
        case DUCKDB_V2_LOGICAL_TYPE_ID_INTEGER:
            return nb::int_(get(duckdb_v2_value_get_int, (int32_t)0));
        case DUCKDB_V2_LOGICAL_TYPE_ID_BIGINT:
            return nb::int_(get(duckdb_v2_value_get_bigint, (int64_t)0));
        case DUCKDB_V2_LOGICAL_TYPE_ID_UTINYINT:
            return nb::int_(get(duckdb_v2_value_get_utinyint, (uint8_t)0));
        case DUCKDB_V2_LOGICAL_TYPE_ID_USMALLINT:
            return nb::int_(get(duckdb_v2_value_get_usmallint, (uint16_t)0));
        case DUCKDB_V2_LOGICAL_TYPE_ID_UINTEGER:
            return nb::int_(get(duckdb_v2_value_get_uint, (uint32_t)0));
        case DUCKDB_V2_LOGICAL_TYPE_ID_UBIGINT:
            return nb::int_(get(duckdb_v2_value_get_ubigint, (uint64_t)0));
        case DUCKDB_V2_LOGICAL_TYPE_ID_FLOAT:
        case DUCKDB_V2_LOGICAL_TYPE_ID_DOUBLE:
            return nb::float_(get(duckdb_v2_value_get_double, 0.0));
        case DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR: {
            const char* text = nullptr;
            idx_t length = 0;
            e.check(duckdb_v2_value_get_varchar(val, &text, &length, e.out()));
            return nb::str(text, length);
        }
        default: {
            duckdb_v2_logical_type_handle varchar = nullptr;
            e.check(duckdb_v2_factory_create_type_from_id(
                factory, DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR, nullptr, nullptr, nullptr, 0, &varchar,
                e.out()));
            duckdb_v2_value_handle cast = nullptr;
            code = duckdb_v2_value_cast(factory, val, varchar, &cast, e.out());
            duckdb_v2_logical_type_destroy(&varchar);
            e.check(code);
            nb::object out;
            try {
                out = value_to_python(factory, cast);
            } catch (...) {
                duckdb_v2_value_destroy(&cast);
                throw;
            }
            duckdb_v2_value_destroy(&cast);
            return out;
        }
    }
}

bool table_output_type_supported(DUCKDB_V2_LOGICAL_TYPE_ID t) {
    switch (t) {
        case DUCKDB_V2_LOGICAL_TYPE_ID_BOOLEAN:
        case DUCKDB_V2_LOGICAL_TYPE_ID_TINYINT:
        case DUCKDB_V2_LOGICAL_TYPE_ID_SMALLINT:
        case DUCKDB_V2_LOGICAL_TYPE_ID_INTEGER:
        case DUCKDB_V2_LOGICAL_TYPE_ID_BIGINT:
        case DUCKDB_V2_LOGICAL_TYPE_ID_UTINYINT:
        case DUCKDB_V2_LOGICAL_TYPE_ID_USMALLINT:
        case DUCKDB_V2_LOGICAL_TYPE_ID_UINTEGER:
        case DUCKDB_V2_LOGICAL_TYPE_ID_UBIGINT:
        case DUCKDB_V2_LOGICAL_TYPE_ID_FLOAT:
        case DUCKDB_V2_LOGICAL_TYPE_ID_DOUBLE:
        case DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR:
        case DUCKDB_V2_LOGICAL_TYPE_ID_BLOB:
        case DUCKDB_V2_LOGICAL_TYPE_ID_DATE:
        case DUCKDB_V2_LOGICAL_TYPE_ID_TIME:
        case DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP:
        case DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_SEC:
        case DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_MS:
        case DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_NS:
        case DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ:
            return true;
        default:
            return false;
    }
}

// Range-checked narrowing for the integer column writes below: a Python 300
// written to a TINYINT column must raise, not silently wrap to 44.
int64_t checked_signed(nb::handle val, int64_t lo, int64_t hi, const char* type) {
    int64_t v = nb::cast<int64_t>(val);
    if (v < lo || v > hi) {
        throw DuckyError("ducky: value " + std::to_string(v) + " out of range for " + type);
    }
    return v;
}

uint64_t checked_unsigned(nb::handle val, uint64_t hi, const char* type) {
    uint64_t v = nb::cast<uint64_t>(val);
    if (v > hi) {
        throw DuckyError("ducky: value " + std::to_string(v) + " out of range for " + type);
    }
    return v;
}

// Floor division, so instants before the epoch round towards the past.
int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    return (a % b != 0 && (a < 0) != (b < 0)) ? q - 1 : q;
}

int64_t micros_to_nanos(int64_t micros) {
    if (micros > INT64_MAX / 1000 || micros < INT64_MIN / 1000) {
        throw DuckyError("ducky: datetime is out of the range of TIMESTAMP_NS");
    }
    return micros * 1000;
}

struct OutColumn {
    duckdb_v2_vector_handle vector = nullptr;
    void* data = nullptr;
    DUCKDB_V2_LOGICAL_TYPE_ID type = DUCKDB_V2_LOGICAL_TYPE_ID_INVALID;
    uint64_t* validity = nullptr;            // taken on the first NULL
    duckdb_v2_arena_handle arena = nullptr;  // taken on the first long string
};

// Sized explicitly, since the bytes may include NULs.
void write_bytes(OutColumn& col, idx_t row, const char* bytes, size_t size) {
    if (size > UINT32_MAX) throw DuckyError("ducky: string value is too long");
    duckdb_v2_bytes& out = ((duckdb_v2_bytes*)col.data)[row];
    if (size <= DUCKDB_V2_BYTES_INLINE_LENGTH) {
        out.value.inlined.length = (uint32_t)size;
        std::memset(out.value.inlined.inlined, 0, sizeof(out.value.inlined.inlined));
        std::memcpy(out.value.inlined.inlined, bytes, size);
        return;
    }
    V2Error e;
    if (!col.arena) e.check(duckdb_v2_vector_get_arena(col.vector, &col.arena, e.out()));
    uint8_t* stored = nullptr;
    e.check(duckdb_v2_arena_allocate(col.arena, size, &stored, e.out()));
    std::memcpy(stored, bytes, size);
    out.value.pointer.length = (uint32_t)size;
    std::memcpy(out.value.pointer.prefix, bytes, sizeof(out.value.pointer.prefix));
    out.value.pointer.ptr = (char*)stored;
}

void write_column_value(OutColumn& col, idx_t row, nb::handle val) {
    if (val.is_none()) {
        if (!col.validity) {
            V2Error e;
            e.check(duckdb_v2_vector_flat_get_validity_mutable(col.vector, &col.validity, e.out()));
        }
        col.validity[row / 64] &= ~((uint64_t)1 << (row % 64));
        return;
    }

    if (col.type == DUCKDB_V2_LOGICAL_TYPE_ID_BLOB) {
        nb::bytes b = nb::cast<nb::bytes>(val);
        write_bytes(col, row, b.c_str(), b.size());
        return;
    }
    if (col.type == DUCKDB_V2_LOGICAL_TYPE_ID_VARCHAR) {
        // PyUnicode's cached UTF-8 buffer saves a copy.
        if (!nb::isinstance<nb::str>(val)) {
            throw DuckyError("ducky: expected str for VARCHAR column");
        }
        Py_ssize_t size = 0;
        const char* s = PyUnicode_AsUTF8AndSize(val.ptr(), &size);
        if (!s) throw nb::python_error();
        write_bytes(col, row, s, (size_t)size);
        return;
    }

    void* data = col.data;
    switch (col.type) {
        case DUCKDB_V2_LOGICAL_TYPE_ID_BOOLEAN:
            ((bool*)data)[row] = nb::cast<bool>(val);
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_TINYINT:
            ((int8_t*)data)[row] = (int8_t)checked_signed(val, INT8_MIN, INT8_MAX, "TINYINT");
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_SMALLINT:
            ((int16_t*)data)[row] = (int16_t)checked_signed(val, INT16_MIN, INT16_MAX, "SMALLINT");
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_INTEGER:
            ((int32_t*)data)[row] = (int32_t)checked_signed(val, INT32_MIN, INT32_MAX, "INTEGER");
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_BIGINT:
            ((int64_t*)data)[row] = nb::cast<int64_t>(val);
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_UTINYINT:
            ((uint8_t*)data)[row] = (uint8_t)checked_unsigned(val, UINT8_MAX, "UTINYINT");
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_USMALLINT:
            ((uint16_t*)data)[row] = (uint16_t)checked_unsigned(val, UINT16_MAX, "USMALLINT");
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_UINTEGER:
            ((uint32_t*)data)[row] = (uint32_t)checked_unsigned(val, UINT32_MAX, "UINTEGER");
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_UBIGINT:
            ((uint64_t*)data)[row] = nb::cast<uint64_t>(val);
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_FLOAT:
            ((float*)data)[row] = (float)nb::cast<double>(val);
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_DOUBLE:
            ((double*)data)[row] = nb::cast<double>(val);
            break;
        // Temporal columns accept the matching datetime object or the raw
        // integer storage value (days for DATE, the timestamp's native unit
        // otherwise). datetime.datetime is a date subclass, so DATE excludes it
        // rather than silently truncating the time part.
        case DUCKDB_V2_LOGICAL_TYPE_ID_DATE: {
            PyObject* raw = val.ptr();
            if (PyDate_Check(raw) && !PyDateTime_Check(raw)) {
                ((int32_t*)data)[row] = pydt::date_to_duckdb(raw).days;
            } else {
                ((int32_t*)data)[row] = (int32_t)checked_signed(val, INT32_MIN, INT32_MAX, "DATE");
            }
            break;
        }
        case DUCKDB_V2_LOGICAL_TYPE_ID_TIME: {
            PyObject* raw = val.ptr();
            if (PyTime_Check(raw)) {
                ((int64_t*)data)[row] = pydt::time_to_duckdb(raw).micros;
            } else {
                ((int64_t*)data)[row] = nb::cast<int64_t>(val);
            }
            break;
        }
        case DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP:
        case DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_TZ:
            ((int64_t*)data)[row] = PyDateTime_Check(val.ptr()) ? pydt::datetime_to_micros(val)
                                                                : nb::cast<int64_t>(val);
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_SEC:
            ((int64_t*)data)[row] = PyDateTime_Check(val.ptr())
                                        ? floor_div(pydt::datetime_to_micros(val), 1000000)
                                        : nb::cast<int64_t>(val);
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_MS:
            ((int64_t*)data)[row] = PyDateTime_Check(val.ptr())
                                        ? floor_div(pydt::datetime_to_micros(val), 1000)
                                        : nb::cast<int64_t>(val);
            break;
        case DUCKDB_V2_LOGICAL_TYPE_ID_TIMESTAMP_NS:
            ((int64_t*)data)[row] = PyDateTime_Check(val.ptr())
                                        ? micros_to_nanos(pydt::datetime_to_micros(val))
                                        : nb::cast<int64_t>(val);
            break;
        default:
            // Unreachable: create_table_function rejects unsupported column
            // types at registration (table_output_type_supported).
            throw DuckyError("ducky: table function column type is not supported");
    }
}

void table_bind(duckdb_v2_function_bind_info_handle info,
                duckdb_v2_table_function_bind_info_handle result, duckdb_v2_context_handle context,
                duckdb_v2_error_info_handle* err) {
    nb::gil_scoped_acquire gil;
    report(err, [&] {
        V2Error e;
        void* user_data = nullptr;
        e.check(duckdb_v2_function_bind_get_user_data(info, &user_data, e.out()));
        auto* ctx = (TableUDFContext*)user_data;
        duckdb_v2_factory_handle factory = nullptr;
        e.check(duckdb_v2_context_get_factory(context, &factory, e.out()));

        idx_t n_params = 0;
        e.check(duckdb_v2_function_bind_get_arg_count(info, &n_params, nullptr, nullptr, nullptr,
                                                      e.out()));
        nb::list python_args;
        for (idx_t i = 0; i < n_params; ++i) {
            duckdb_v2_value_handle val = nullptr;
            e.check(duckdb_v2_function_bind_get_arg_value(info, i, &val, e.out()));
            nb::object py_val;
            try {
                py_val = value_to_python(factory, val);
            } catch (...) {
                duckdb_v2_value_destroy(&val);
                throw;
            }
            duckdb_v2_value_destroy(&val);
            python_args.append(py_val);
        }

        for (size_t i = 0; i < ctx->col_names.size(); ++i) {
            const std::string& col_name = ctx->col_names[i];
            e.check(duckdb_v2_table_function_bind_add_result_column(
                result, col_name.data(), col_name.size(), ctx->col_types[i], e.out()));
        }

        auto bd = std::make_unique<TableUDFBindData>(TableUDFBindData{ctx->factory, python_args});
        duckdb_v2_opaque data{bd.get(), &gil_delete<TableUDFBindData>, nullptr};
        e.check(duckdb_v2_function_bind_set_bind_data(info, &data, e.out()));
        // NOLINTNEXTLINE(bugprone-unused-return-value)
        (void)bd.release();
    });
}

void table_init(duckdb_v2_table_function_init_global_info_handle info,
                duckdb_v2_context_handle context, duckdb_v2_error_info_handle* err) {
    nb::gil_scoped_acquire gil;
    report(err, [&] {
        V2Error e;
        void* bind_data = nullptr;
        e.check(duckdb_v2_table_function_init_global_get_bind_data(info, &bind_data, e.out()));
        auto* bd = (TableUDFBindData*)bind_data;
        // Single-threaded: Python generators are not thread-safe.
        e.check(duckdb_v2_table_function_init_global_set_max_threads(info, 1, e.out()));
        // The scan calls PyIter_Next, which needs a real iterator: a factory
        // may return any iterable (a list, say).
        nb::object produced = bd->factory(*nb::tuple(bd->python_args));
        PyObject* iterator = PyObject_GetIter(produced.ptr());
        if (!iterator) throw nb::python_error();
        nb::object gen = nb::steal(iterator);
        auto id = std::make_unique<TableUDFInitData>(TableUDFInitData{std::move(gen)});
        duckdb_v2_opaque data{id.get(), &gil_delete<TableUDFInitData>, nullptr};
        e.check(duckdb_v2_table_function_init_global_set_global_state(info, &data, e.out()));
        // NOLINTNEXTLINE(bugprone-unused-return-value)
        (void)id.release();
    });
}

void table_main(duckdb_v2_table_function_exec_info_handle info, duckdb_v2_context_handle context,
                duckdb_v2_error_info_handle* err) {
    nb::gil_scoped_acquire gil;
    report(err, [&] {
        V2Error e;
        void* user_data = nullptr;
        void* global_state = nullptr;
        e.check(duckdb_v2_table_function_exec_get_user_data(info, &user_data, e.out()));
        e.check(duckdb_v2_table_function_exec_get_global_state(info, &global_state, e.out()));
        auto* ctx = (TableUDFContext*)user_data;
        auto* id = (TableUDFInitData*)global_state;
        if (id->exhausted) return;  // an empty output ends the scan

        duckdb_v2_data_chunk_handle output = nullptr;
        e.check(duckdb_v2_table_function_exec_get_output_chunk(info, &output, e.out()));
        size_t n_cols = ctx->col_type_ids.size();
        std::vector<OutColumn> cols(n_cols);
        for (size_t c = 0; c < n_cols; ++c) {
            OutColumn& col = cols[c];
            col.type = ctx->col_type_ids[c];
            e.check(duckdb_v2_data_chunk_get_vector(output, c, &col.vector, e.out()));
            e.check(duckdb_v2_vector_get_data_mutable(col.vector, &col.data, e.out()));
        }

        idx_t rows = 0;
        idx_t vec_size = duckdb_vector_size();
        PyObject* gen = id->generator.ptr();
        while (rows < vec_size) {
            // PyIter_Next avoids a `builtins.next` lookup and a StopIteration
            // per row. NULL without an error set means exhaustion.
            PyObject* raw = PyIter_Next(gen);
            if (!raw) {
                if (PyErr_Occurred()) throw nb::python_error();
                id->exhausted = true;
                break;
            }
            nb::object row_obj = nb::steal(raw);

            // Accept single-value non-tuple rows as a 1-element tuple.
            nb::tuple row;
            if (nb::isinstance<nb::tuple>(row_obj)) {
                row = nb::cast<nb::tuple>(row_obj);
            } else {
                row = nb::make_tuple(row_obj);
            }
            // row[c] below is unchecked (PyTuple_GET_ITEM), so a wrong-arity
            // row must be rejected here, not discovered by reading past the
            // tuple's end.
            if (nb::len(row) != n_cols) {
                throw DuckyError("ducky: table function row has " + std::to_string(nb::len(row)) +
                                 " values; expected " + std::to_string(n_cols));
            }

            for (size_t c = 0; c < n_cols; ++c) write_column_value(cols[c], rows, row[c]);
            ++rows;
        }
        // The engine takes the first vector's size as the chunk's.
        if (rows > 0) e.check(duckdb_v2_vector_set_size(cols[0].vector, rows, e.out()));
    });
}

struct ParsedTypes {
    std::vector<duckdb_v2_logical_type_handle> types;
    ~ParsedTypes() {
        for (duckdb_v2_logical_type_handle& t : types) duckdb_v2_logical_type_destroy(&t);
    }
};

}  // namespace

void create_table_function(Connection& con, const std::string& name, nb::callable factory,
                           nb::object parameters, nb::object columns) {
    // PyDateTimeAPI is static per translation unit, and the trampolines'
    // temporal branches use it. Registration always precedes execution.
    if (!PyDateTimeAPI) PyDateTime_IMPORT;
    std::shared_ptr<DuckDBHandle> handle = con.open_handle();
    duckdb_v2_connection_handle connection = handle->connection_v2;
    V2Error err;
    duckdb_v2_factory_handle types = nullptr;
    err.check(duckdb_v2_connection_get_factory(connection, &types, err.out()));
    auto ctx = std::make_unique<TableUDFContext>();
    ctx->factory = handle->keep(factory);

    if (!nb::isinstance<nb::sequence>(parameters)) {
        throw DuckyError("ducky: table function '" + name + "': parameters must be a list");
    }
    ParsedTypes param_types;
    for (auto item : nb::cast<nb::sequence>(parameters)) {
        param_types.types.push_back(v2_parse_type(types, nb::cast<std::string>(item)));
    }

    if (!nb::isinstance<nb::dict>(columns)) {
        throw DuckyError("ducky: table function '" + name + "': columns must be a dict");
    }
    nb::dict cols = nb::cast<nb::dict>(columns);
    if (cols.size() == 0) {
        throw DuckyError("ducky: table function '" + name + "' must declare at least one column");
    }
    for (auto item : cols) {
        std::string col_name = nb::cast<std::string>(item.first);
        duckdb_v2_logical_type_handle lt = v2_parse_type(types, nb::cast<std::string>(item.second));
        // Push before validating so ~TableUDFContext owns (and frees) lt on throw.
        ctx->col_names.push_back(col_name);
        ctx->col_types.push_back(lt);
        DUCKDB_V2_LOGICAL_TYPE_ID tid = DUCKDB_V2_LOGICAL_TYPE_ID_INVALID;
        err.check(duckdb_v2_logical_type_get_id(lt, &tid, err.out()));
        ctx->col_type_ids.push_back(tid);
        if (!table_output_type_supported(tid)) {
            throw DuckyError("ducky: table function '" + name + "': column '" + col_name +
                             "' has unsupported type " + v2_type_name(lt) +
                             " (supported: BOOLEAN, [U]TINYINT..BIGINT, FLOAT, DOUBLE, VARCHAR, "
                             "BLOB, DATE, TIME, TIMESTAMP[_S/MS/NS/TZ])");
        }
    }

    const std::string failed = "ducky: failed to register table function '" + name + "': ";
    auto check = [&](DUCKDB_V2_ERROR code) { err.check(code, failed); };
    duckdb_v2_table_function_handle f = nullptr;
    check(duckdb_v2_table_function_create(types, &f, err.out()));
    std::unique_ptr<duckdb_v2_table_function_handle, void (*)(duckdb_v2_table_function_handle*)>
        f_guard(&f, [](duckdb_v2_table_function_handle* p) {
            // A failed registration's function drops the last reference to the
            // context, whose destructor takes the GIL.
            release_gil_if_held release;
            duckdb_v2_table_function_destroy(p);
        });
    check(duckdb_v2_table_function_set_name(f, name.data(), name.size(), err.out()));
    duckdb_v2_function_signature_handle sig = nullptr;
    check(duckdb_v2_table_function_get_signature(f, &sig, err.out()));
    for (size_t i = 0; i < param_types.types.size(); ++i) {
        std::string param = "arg" + std::to_string(i + 1);
        check(duckdb_v2_function_signature_add_parameter(
            sig, param.data(), param.size(), param_types.types[i], nullptr,
            DUCKDB_V2_FUNCTION_PARAMETER_KIND_POSITIONAL_ONLY, err.out()));
    }
    // Ownership of ctx passes to the function object here: its destroy callback
    // runs whether or not registration succeeds.
    duckdb_v2_opaque data{ctx.get(), &gil_delete<TableUDFContext>, nullptr};
    check(duckdb_v2_table_function_set_user_data(f, &data, err.out()));
    // NOLINTNEXTLINE(bugprone-unused-return-value)
    (void)ctx.release();
    check(duckdb_v2_table_function_set_bind_callback(f, &table_bind, err.out()));
    check(duckdb_v2_table_function_set_init_global_callback(f, &table_init, err.out()));
    check(duckdb_v2_table_function_set_exec_callback(f, &table_main, err.out()));
    check(duckdb_v2_table_function_set_projection_pushdown(f, false, err.out()));

    DUCKDB_V2_ERROR code;
    {
        // Registration takes the client context lock, which a query running
        // Python callbacks on another thread may hold while it waits for the
        // GIL.
        release_gil_if_held release;
        code = duckdb_v2_connection_register_table_function(connection, f, err.out());
    }
    check(code);
}
