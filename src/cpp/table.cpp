#include "table.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <cstring>

#include "connection.hpp"
#include "ducky.hpp"
#include "function.hpp"
#include "py_datetime.hpp"

namespace nb = nanobind;

namespace {

// Convert a duckdb_value (bound SQL parameter) to a Python object.
nb::object duckdb_value_to_python(duckdb_value val) {
    if (duckdb_is_null_value(val)) return nb::none();
    // duckdb_get_value_type returns a borrowed pointer — do not destroy.
    duckdb_type tid = duckdb_get_type_id(duckdb_get_value_type(val));

    switch (tid) {
        case DUCKDB_TYPE_BOOLEAN:
            return nb::bool_(duckdb_get_bool(val));
        case DUCKDB_TYPE_TINYINT:
            return nb::int_((int8_t)duckdb_get_int8(val));
        case DUCKDB_TYPE_SMALLINT:
            return nb::int_((int16_t)duckdb_get_int16(val));
        case DUCKDB_TYPE_INTEGER:
            return nb::int_((int32_t)duckdb_get_int32(val));
        case DUCKDB_TYPE_BIGINT:
            return nb::int_((int64_t)duckdb_get_int64(val));
        case DUCKDB_TYPE_UTINYINT:
            return nb::int_((uint8_t)duckdb_get_uint8(val));
        case DUCKDB_TYPE_USMALLINT:
            return nb::int_((uint16_t)duckdb_get_uint16(val));
        case DUCKDB_TYPE_UINTEGER:
            return nb::int_((uint32_t)duckdb_get_uint32(val));
        case DUCKDB_TYPE_UBIGINT:
            return nb::int_((uint64_t)duckdb_get_uint64(val));
        case DUCKDB_TYPE_FLOAT:
        case DUCKDB_TYPE_DOUBLE:
            return nb::float_(duckdb_get_double(val));
        default: {
            char* s = duckdb_get_varchar(val);
            nb::object result = nb::str(s);
            duckdb_free(s);
            return result;
        }
    }
}

// Column types supported by write_column_value.
bool table_output_type_supported(duckdb_type t) {
    switch (t) {
        case DUCKDB_TYPE_BOOLEAN:
        case DUCKDB_TYPE_TINYINT:
        case DUCKDB_TYPE_SMALLINT:
        case DUCKDB_TYPE_INTEGER:
        case DUCKDB_TYPE_BIGINT:
        case DUCKDB_TYPE_UTINYINT:
        case DUCKDB_TYPE_USMALLINT:
        case DUCKDB_TYPE_UINTEGER:
        case DUCKDB_TYPE_UBIGINT:
        case DUCKDB_TYPE_FLOAT:
        case DUCKDB_TYPE_DOUBLE:
        case DUCKDB_TYPE_VARCHAR:
        case DUCKDB_TYPE_BLOB:
        case DUCKDB_TYPE_DATE:
        case DUCKDB_TYPE_TIME:
        case DUCKDB_TYPE_TIMESTAMP:
        case DUCKDB_TYPE_TIMESTAMP_S:
        case DUCKDB_TYPE_TIMESTAMP_MS:
        case DUCKDB_TYPE_TIMESTAMP_NS:
        case DUCKDB_TYPE_TIMESTAMP_TZ:
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

// Write a Python scalar value into a duckdb_vector at position `row`.
void write_column_value(duckdb_vector vec, idx_t row, nb::object val, duckdb_type tid) {
    if (val.is_none()) {
        duckdb_vector_ensure_validity_writable(vec);
        uint64_t* validity = duckdb_vector_get_validity(vec);
        duckdb_validity_set_row_invalid(validity, row);
        return;
    }

    if (tid == DUCKDB_TYPE_BLOB) {
        // Length-aware assignment: blobs (and Python strings) may carry
        // embedded NUL bytes, which the strlen-based variant would truncate at.
        nb::bytes b = nb::cast<nb::bytes>(val);
        duckdb_vector_assign_string_element_len(vec, row, b.c_str(), b.size());
        return;
    }
    if (tid == DUCKDB_TYPE_VARCHAR) {
        // Borrow PyUnicode's cached UTF-8 buffer; no std::string copy.
        if (!nb::isinstance<nb::str>(val)) {
            throw DuckyError("ducky: expected str for VARCHAR column");
        }
        Py_ssize_t size = 0;
        const char* s = PyUnicode_AsUTF8AndSize(val.ptr(), &size);
        if (!s) throw nb::python_error();
        duckdb_vector_assign_string_element_len(vec, row, s, (idx_t)size);
        return;
    }

    void* data = duckdb_vector_get_data(vec);

    switch (tid) {
        case DUCKDB_TYPE_BOOLEAN:
            ((bool*)data)[row] = nb::cast<bool>(val);
            break;
        case DUCKDB_TYPE_TINYINT:
            ((int8_t*)data)[row] = (int8_t)checked_signed(val, INT8_MIN, INT8_MAX, "TINYINT");
            break;
        case DUCKDB_TYPE_SMALLINT:
            ((int16_t*)data)[row] = (int16_t)checked_signed(val, INT16_MIN, INT16_MAX, "SMALLINT");
            break;
        case DUCKDB_TYPE_INTEGER:
            ((int32_t*)data)[row] = (int32_t)checked_signed(val, INT32_MIN, INT32_MAX, "INTEGER");
            break;
        case DUCKDB_TYPE_BIGINT:
            ((int64_t*)data)[row] = nb::cast<int64_t>(val);
            break;
        case DUCKDB_TYPE_UTINYINT:
            ((uint8_t*)data)[row] = (uint8_t)checked_unsigned(val, UINT8_MAX, "UTINYINT");
            break;
        case DUCKDB_TYPE_USMALLINT:
            ((uint16_t*)data)[row] = (uint16_t)checked_unsigned(val, UINT16_MAX, "USMALLINT");
            break;
        case DUCKDB_TYPE_UINTEGER:
            ((uint32_t*)data)[row] = (uint32_t)checked_unsigned(val, UINT32_MAX, "UINTEGER");
            break;
        case DUCKDB_TYPE_UBIGINT:
            ((uint64_t*)data)[row] = nb::cast<uint64_t>(val);
            break;
        case DUCKDB_TYPE_FLOAT:
            ((float*)data)[row] = (float)nb::cast<double>(val);
            break;
        case DUCKDB_TYPE_DOUBLE:
            ((double*)data)[row] = nb::cast<double>(val);
            break;
        // Temporal columns accept the matching datetime object or the raw
        // integer storage value (days for DATE, the timestamp's native unit
        // otherwise). datetime.datetime is a date subclass, so DATE excludes it
        // rather than silently truncating the time part.
        case DUCKDB_TYPE_DATE: {
            PyObject* raw = val.ptr();
            if (PyDate_Check(raw) && !PyDateTime_Check(raw)) {
                ((int32_t*)data)[row] = pydt::date_to_duckdb(raw).days;
            } else {
                ((int32_t*)data)[row] = (int32_t)checked_signed(val, INT32_MIN, INT32_MAX, "DATE");
            }
            break;
        }
        case DUCKDB_TYPE_TIME: {
            PyObject* raw = val.ptr();
            if (PyTime_Check(raw)) {
                ((int64_t*)data)[row] = pydt::time_to_duckdb(raw).micros;
            } else {
                ((int64_t*)data)[row] = nb::cast<int64_t>(val);
            }
            break;
        }
        case DUCKDB_TYPE_TIMESTAMP:
        case DUCKDB_TYPE_TIMESTAMP_TZ:
            ((int64_t*)data)[row] = PyDateTime_Check(val.ptr()) ? pydt::datetime_to_micros(val)
                                                                : nb::cast<int64_t>(val);
            break;
        case DUCKDB_TYPE_TIMESTAMP_S:
            ((int64_t*)data)[row] = PyDateTime_Check(val.ptr())
                                        ? floor_div(pydt::datetime_to_micros(val), 1000000)
                                        : nb::cast<int64_t>(val);
            break;
        case DUCKDB_TYPE_TIMESTAMP_MS:
            ((int64_t*)data)[row] = PyDateTime_Check(val.ptr())
                                        ? floor_div(pydt::datetime_to_micros(val), 1000)
                                        : nb::cast<int64_t>(val);
            break;
        case DUCKDB_TYPE_TIMESTAMP_NS:
            ((int64_t*)data)[row] = PyDateTime_Check(val.ptr())
                                        ? micros_to_nanos(pydt::datetime_to_micros(val))
                                        : nb::cast<int64_t>(val);
            break;
        default:
            // Unreachable: create_table_function rejects unsupported column
            // types at registration (table_output_type_supported).
            throw DuckyError(std::string("ducky: table function column type ") +
                             duckdb_type_name(tid) + " is not supported");
    }
}

void table_bind(duckdb_bind_info info) {
    auto* ctx = (TableUDFContext*)duckdb_bind_get_extra_info(info);
    nb::gil_scoped_acquire gil;
    guard(
        [&] {
            idx_t n_params = duckdb_bind_get_parameter_count(info);
            nb::list python_args;
            for (idx_t i = 0; i < n_params; ++i) {
                duckdb_value val = duckdb_bind_get_parameter(info, i);
                nb::object py_val;
                try {
                    py_val = duckdb_value_to_python(val);
                } catch (...) {
                    duckdb_destroy_value(&val);
                    throw;
                }
                duckdb_destroy_value(&val);
                python_args.append(py_val);
            }

            for (idx_t i = 0; i < (idx_t)ctx->col_names.size(); ++i) {
                duckdb_bind_add_result_column(info, ctx->col_names[i].c_str(), ctx->col_types[i]);
            }

            auto* bd =
                new TableUDFBindData{ctx->factory, python_args, ctx->col_type_ids, ctx->col_names};
            duckdb_bind_set_bind_data(info, bd, &gil_delete<TableUDFBindData>);
        },
        [&](const char* what) { duckdb_bind_set_error(info, what); });
}

void table_init(duckdb_init_info info) {
    auto* bd = (TableUDFBindData*)duckdb_init_get_bind_data(info);
    // Single-threaded: Python generators are not thread-safe.
    duckdb_init_set_max_threads(info, 1);
    nb::gil_scoped_acquire gil;
    guard(
        [&] {
            nb::object gen = bd->factory(*nb::tuple(bd->python_args));
            auto* id = new TableUDFInitData{std::move(gen)};
            duckdb_init_set_init_data(info, id, &gil_delete<TableUDFInitData>);
        },
        [&](const char* what) { duckdb_init_set_error(info, what); });
}

void table_main(duckdb_function_info info, duckdb_data_chunk output) {
    auto* bd = (TableUDFBindData*)duckdb_function_get_bind_data(info);
    auto* id = (TableUDFInitData*)duckdb_function_get_init_data(info);

    if (id->exhausted) {
        duckdb_data_chunk_set_size(output, 0);
        return;
    }

    nb::gil_scoped_acquire gil;
    idx_t n_cols = (idx_t)bd->col_type_ids.size();
    idx_t rows = 0;
    idx_t vec_size = duckdb_vector_size();

    bool ok = guard(
        [&] {
            PyObject* gen = id->generator.ptr();
            while (rows < vec_size) {
                // PyIter_Next: returns NULL at exhaustion (no error) or NULL +
                // error set on failure. Avoids a `builtins.next` attr lookup
                // and a StopIteration throw per row.
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
                if (nb::len(row) != (size_t)n_cols) {
                    throw DuckyError("ducky: table function row has " +
                                     std::to_string(nb::len(row)) + " values; expected " +
                                     std::to_string(n_cols));
                }

                for (idx_t c = 0; c < n_cols; ++c) {
                    duckdb_vector vec = duckdb_data_chunk_get_vector(output, c);
                    write_column_value(vec, rows, row[c], bd->col_type_ids[c]);
                }
                ++rows;
            }
        },
        [&](const char* what) { duckdb_function_set_error(info, what); });
    if (!ok) return;

    duckdb_data_chunk_set_size(output, rows);
}

}  // namespace

void create_table_function(Connection& con, const std::string& name, nb::callable factory,
                           nb::object parameters, nb::object columns) {
    // PyDateTimeAPI is static per TU; prime ours so the temporal branches in
    // write_column_value / pydt:: helpers work when the trampolines run.
    // Registration always precedes execution, so priming here suffices.
    if (!PyDateTimeAPI) PyDateTime_IMPORT;
    std::shared_ptr<DuckDBHandle> handle = con.open_handle();
    duckdb_connection raw = handle->connection;
    TypeParser types(handle->database, raw);
    auto ctx = std::make_unique<TableUDFContext>();
    ctx->factory = handle->keep(factory);

    // Parse parameter types.
    if (!nb::isinstance<nb::sequence>(parameters)) {
        throw DuckyError("ducky: table function '" + name + "': parameters must be a list");
    }
    nb::sequence pseq = nb::cast<nb::sequence>(parameters);
    for (auto item : pseq) {
        ctx->param_types.push_back(types.parse(nb::cast<std::string>(item)));
    }

    // Parse output columns from dict {name: type}.
    if (!nb::isinstance<nb::dict>(columns)) {
        throw DuckyError("ducky: table function '" + name + "': columns must be a dict");
    }
    nb::dict cols = nb::cast<nb::dict>(columns);
    if (cols.size() == 0) {
        throw DuckyError("ducky: table function '" + name + "' must declare at least one column");
    }
    for (auto item : cols) {
        std::string col_name = nb::cast<std::string>(item.first);
        std::string type_str = nb::cast<std::string>(item.second);
        duckdb_logical_type lt = types.parse(type_str);
        // Push before validating so ~TableUDFContext owns (and frees) lt on throw.
        ctx->col_names.push_back(col_name);
        ctx->col_types.push_back(lt);
        ctx->col_type_ids.push_back(duckdb_get_type_id(lt));
        if (!table_output_type_supported(ctx->col_type_ids.back())) {
            throw DuckyError("ducky: table function '" + name + "': column '" + col_name +
                             "' has unsupported type " +
                             duckdb_type_name(ctx->col_type_ids.back()) +
                             " (supported: BOOLEAN, [U]TINYINT..BIGINT, FLOAT, DOUBLE, VARCHAR, "
                             "BLOB, DATE, TIME, TIMESTAMP[_S/MS/NS/TZ])");
        }
    }

    duckdb_table_function f = duckdb_create_table_function();
    duckdb_table_function_set_name(f, name.c_str());
    for (duckdb_logical_type t : ctx->param_types) {
        duckdb_table_function_add_parameter(f, t);
    }
    // Ownership of ctx passes to the function object here: its destroy callback
    // runs whether or not registration succeeds (a failed registration's
    // duckdb_destroy_table_function drops the last reference).
    duckdb_table_function_set_extra_info(f, ctx.get(), &gil_delete<TableUDFContext>);
    // NOLINTNEXTLINE(bugprone-unused-return-value)
    (void)ctx.release();
    duckdb_table_function_set_bind(f, &table_bind);
    duckdb_table_function_set_init(f, &table_init);
    duckdb_table_function_set_function(f, &table_main);
    duckdb_table_function_supports_projection_pushdown(f, false);

    duckdb_state state;
    {
        // Registration takes the client context lock, which a query running
        // Python callbacks on another thread may hold while it waits for the
        // GIL; a failed registration's destructor takes the GIL itself.
        release_gil_if_held release;
        state = duckdb_register_table_function(raw, f);
        duckdb_destroy_table_function(&f);
    }
    if (state == DuckDBError) {
        throw DuckyError("ducky: failed to register table function '" + name + "'");
    }
}
