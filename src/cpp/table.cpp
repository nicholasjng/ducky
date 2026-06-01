#include "table.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstring>

#include "connection.hpp"
#include "ducky.hpp"
#include "function.hpp"

namespace nb = nanobind;

namespace {

// Convert a duckdb_value (bound SQL parameter) to a Python object.
nb::object duckdb_value_to_python(duckdb_value val) {
    // duckdb_get_value_type returns a BORROWED reference (pointer to val's own field).
    // Do NOT call duckdb_destroy_logical_type on it.
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

// Write a Python scalar value into a duckdb_vector at position `row`.
void write_column_value(duckdb_vector vec, idx_t row, nb::object val, duckdb_type tid) {
    if (val.is_none()) {
        duckdb_vector_ensure_validity_writable(vec);
        uint64_t* validity = duckdb_vector_get_validity(vec);
        duckdb_validity_set_row_invalid(validity, row);
        return;
    }

    if (tid == DUCKDB_TYPE_VARCHAR || tid == DUCKDB_TYPE_BLOB) {
        std::string s = nb::cast<std::string>(val);
        duckdb_vector_assign_string_element(vec, row, s.c_str());
        return;
    }

    void* data = duckdb_vector_get_data(vec);

    switch (tid) {
        case DUCKDB_TYPE_BOOLEAN:
            ((bool*)data)[row] = nb::cast<bool>(val);
            break;
        case DUCKDB_TYPE_TINYINT:
            ((int8_t*)data)[row] = (int8_t)nb::cast<int64_t>(val);
            break;
        case DUCKDB_TYPE_SMALLINT:
            ((int16_t*)data)[row] = (int16_t)nb::cast<int64_t>(val);
            break;
        case DUCKDB_TYPE_INTEGER:
            ((int32_t*)data)[row] = (int32_t)nb::cast<int64_t>(val);
            break;
        case DUCKDB_TYPE_BIGINT:
            ((int64_t*)data)[row] = nb::cast<int64_t>(val);
            break;
        case DUCKDB_TYPE_UTINYINT:
            ((uint8_t*)data)[row] = (uint8_t)nb::cast<uint64_t>(val);
            break;
        case DUCKDB_TYPE_USMALLINT:
            ((uint16_t*)data)[row] = (uint16_t)nb::cast<uint64_t>(val);
            break;
        case DUCKDB_TYPE_UINTEGER:
            ((uint32_t*)data)[row] = (uint32_t)nb::cast<uint64_t>(val);
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
        default:
            // For unsupported types fall back to string assignment.
            {
                std::string s = nb::cast<std::string>(nb::str(val));
                duckdb_vector_assign_string_element(vec, row, s.c_str());
            }
            break;
    }
}

void table_extra_info_destroy(void* ptr) {
    if (!ptr) return;
    auto* ctx = (TableUDFContext*)ptr;
    for (duckdb_logical_type& t : ctx->param_types) duckdb_destroy_logical_type(&t);
    for (duckdb_logical_type& t : ctx->col_types) duckdb_destroy_logical_type(&t);
    {
        nb::gil_scoped_acquire gil;
        delete ctx;
    }
}

void bind_data_destroy(void* ptr) {
    if (!ptr) return;
    nb::gil_scoped_acquire gil;
    delete (TableUDFBindData*)ptr;
}

void init_data_destroy(void* ptr) {
    if (!ptr) return;
    nb::gil_scoped_acquire gil;
    delete (TableUDFInitData*)ptr;
}

void table_bind(duckdb_bind_info info) {
    auto* ctx = (TableUDFContext*)duckdb_bind_get_extra_info(info);
    nb::gil_scoped_acquire gil;
    try {
        idx_t n_params = duckdb_bind_get_parameter_count(info);
        nb::list python_args;
        for (idx_t i = 0; i < n_params; ++i) {
            duckdb_value val = duckdb_bind_get_parameter(info, i);
            python_args.append(duckdb_value_to_python(val));
            duckdb_destroy_value(&val);
        }

        for (idx_t i = 0; i < (idx_t)ctx->col_names.size(); ++i) {
            duckdb_bind_add_result_column(info, ctx->col_names[i].c_str(), ctx->col_types[i]);
        }

        auto* bd =
            new TableUDFBindData{ctx->factory, python_args, ctx->col_type_ids, ctx->col_names};
        duckdb_bind_set_bind_data(info, bd, &bind_data_destroy);
    } catch (nb::python_error& e) {
        duckdb_bind_set_error(info, e.what());
    } catch (const std::exception& e) {
        duckdb_bind_set_error(info, e.what());
    }
}

void table_init(duckdb_init_info info) {
    auto* bd = (TableUDFBindData*)duckdb_init_get_bind_data(info);
    // Single-threaded: Python generators are not thread-safe.
    duckdb_init_set_max_threads(info, 1);
    nb::gil_scoped_acquire gil;
    try {
        nb::object gen = bd->factory(*nb::tuple(bd->python_args));
        auto* id = new TableUDFInitData{std::move(gen)};
        duckdb_init_set_init_data(info, id, &init_data_destroy);
    } catch (nb::python_error& e) {
        duckdb_init_set_error(info, e.what());
    } catch (const std::exception& e) {
        duckdb_init_set_error(info, e.what());
    }
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

    try {
        while (rows < 2048) {
            nb::object row_obj;
            try {
                row_obj = nb::object(nb::module_::import_("builtins").attr("next")(id->generator));
            } catch (nb::python_error& e) {
                if (e.matches(PyExc_StopIteration)) {
                    id->exhausted = true;
                    break;
                }
                throw;
            }

            // Accept single-value non-tuple rows as a 1-element tuple.
            nb::tuple row;
            if (nb::isinstance<nb::tuple>(row_obj)) {
                row = nb::cast<nb::tuple>(row_obj);
            } else {
                row = nb::make_tuple(row_obj);
            }

            for (idx_t c = 0; c < n_cols; ++c) {
                duckdb_vector vec = duckdb_data_chunk_get_vector(output, c);
                write_column_value(vec, rows, row[c], bd->col_type_ids[c]);
            }
            ++rows;
        }
    } catch (nb::python_error& e) {
        duckdb_function_set_error(info, e.what());
        return;
    } catch (const std::exception& e) {
        duckdb_function_set_error(info, e.what());
        return;
    }

    duckdb_data_chunk_set_size(output, rows);
}

}  // namespace

void create_table_function(Connection& con, const std::string& name, nb::callable factory,
                           nb::object parameters, nb::object columns) {
    duckdb_connection raw = con.raw_connection();
    auto ctx = std::make_unique<TableUDFContext>();
    ctx->factory = factory;

    // Parse parameter types.
    if (!nb::isinstance<nb::sequence>(parameters)) {
        throw DuckyError("ducky: table function '" + name + "': parameters must be a list");
    }
    nb::sequence pseq = nb::cast<nb::sequence>(parameters);
    for (auto item : pseq) {
        ctx->param_types.push_back(parse_logical_type(raw, nb::cast<std::string>(item)));
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
        duckdb_logical_type lt = parse_logical_type(raw, type_str);
        ctx->col_names.push_back(col_name);
        ctx->col_types.push_back(lt);
        ctx->col_type_ids.push_back(duckdb_get_type_id(lt));
    }

    duckdb_table_function f = duckdb_create_table_function();
    duckdb_table_function_set_name(f, name.c_str());
    for (duckdb_logical_type t : ctx->param_types) {
        duckdb_table_function_add_parameter(f, t);
    }
    duckdb_table_function_set_extra_info(f, ctx.get(), &table_extra_info_destroy);
    duckdb_table_function_set_bind(f, &table_bind);
    duckdb_table_function_set_init(f, &table_init);
    duckdb_table_function_set_function(f, &table_main);
    duckdb_table_function_supports_projection_pushdown(f, false);

    duckdb_state state = duckdb_register_table_function(raw, f);
    duckdb_destroy_table_function(&f);
    if (state == DuckDBError) {
        throw DuckyError("ducky: failed to register table function '" + name + "'");
    }
    (void)ctx.release();
}
