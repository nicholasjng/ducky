#include "appender.hpp"

#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstring>
#include <string>

#include "arrow_abi.h"
#include "arrow_util.hpp"
#include "connection.hpp"
#include "ducky.hpp"
#include "function.hpp"
#include "py_datetime.hpp"

namespace {

// Flat dtype + element size for numeric / temporal DuckDB types, or false for
// non-flat types (VARCHAR, nested, DECIMAL, HUGEINT, INTERVAL).
struct FlatDType {
    nb::dlpack::dtype dtype;
    size_t elem_size;
};

bool flat_dtype_for(duckdb_type t, FlatDType& out) {
    if (const TypeSpec* spec = typespec_for(t)) {
        out.dtype = spec->dtype();
        out.elem_size = spec->size;
        return true;
    }
    return false;
}

// Quote a DuckDB identifier for safe embedding in a SQL string.
std::string quote_ident(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('"');
    for (char c : s) {
        if (c == '"') out.push_back('"');
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

// Names + per-column DEFAULT-ness of the target table's stored columns, read
// from DuckDB's `table_description` introspection API. Generated columns are
// skipped: the appender takes no values for them, so the remaining columns line
// up with the appender's. Called once at construction so callers can use
// append_columns({name: ndarray}) and append_row's `ducky.DEFAULT` sentinel.
struct TableIntrospection {
    std::vector<std::string> names;
    std::vector<bool> has_default;
};

TableIntrospection introspect_table(duckdb_connection con,
                                    const std::optional<std::string>& catalog,
                                    const std::optional<std::string>& schema,
                                    const std::string& table) {
    duckdb_table_description desc = nullptr;
    duckdb_state state;
    {
        // Takes the client context lock (see Appender::Appender).
        release_gil_if_held release;
        if (catalog) {
            state = duckdb_table_description_create_ext(
                con, catalog->c_str(), schema ? schema->c_str() : nullptr, table.c_str(), &desc);
        } else {
            state = duckdb_table_description_create(con, schema ? schema->c_str() : nullptr,
                                                    table.c_str(), &desc);
        }
    }
    // desc must be destroyed even on error (per the C API contract).
    if (state == DuckDBError) {
        std::string message = "ducky: appender table introspection failed";
        if (desc) {
            const char* m = duckdb_table_description_error(desc);
            if (m) message += std::string(": ") + m;
        }
        duckdb_table_description_destroy(&desc);
        throw DuckyError(message);
    }

    TableIntrospection out;
    idx_t n = duckdb_table_description_get_column_count(desc);
    out.names.reserve(n);
    out.has_default.reserve(n);
    for (idx_t i = 0; i < n; ++i) {
        bool generated = false;
        duckdb_column_is_generated(desc, i, &generated);
        if (generated) continue;
        char* name = duckdb_table_description_get_column_name(desc, i);
        out.names.emplace_back(name ? name : "");
        duckdb_free(name);
        bool def = false;
        duckdb_column_has_default(desc, i, &def);
        out.has_default.push_back(def);
    }
    duckdb_table_description_destroy(&desc);
    return out;
}

// 1-D, C-contiguous ndarray view of unknown dtype; we dtype-check manually
// for better error messages than nanobind's templated reject.
using AnyArray = nb::ndarray<nb::ro, nb::c_contig>;

bool dtype_eq(nb::dlpack::dtype a, nb::dlpack::dtype b) {
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

}  // namespace

Appender::Appender(Connection& connection, std::string table, std::optional<std::string> schema,
                   std::optional<std::string> catalog) {
    // PyDateTimeAPI is static per TU; prime ours so append_row's temporal
    // branches (pydt:: helpers) work. Construction precedes every append.
    if (!PyDateTimeAPI) PyDateTime_IMPORT;
    connection_ = connection.open_handle();
    duckdb_connection con = connection_->connection;

    duckdb_state state;
    {
        // Creating, flushing and closing an appender take the client context
        // lock, which a query running Python callbacks on another thread may
        // hold while it waits for the GIL.
        release_gil_if_held release;
        if (catalog) {
            state = duckdb_appender_create_ext(
                con, catalog->c_str(), schema ? schema->c_str() : nullptr, table.c_str(), &handle_);
        } else {
            state = duckdb_appender_create(con, schema ? schema->c_str() : nullptr, table.c_str(),
                                           &handle_);
        }
    }
    if (state == DuckDBError) {
        std::string message = "ducky: failed to create appender for " + quote_ident(table);
        if (handle_) {
            duckdb_error_data err = duckdb_appender_error_data(handle_);
            if (err && duckdb_error_data_has_error(err)) {
                message += ": ";
                message += duckdb_error_data_message(err);
            }
            duckdb_destroy_error_data(&err);
            duckdb_appender_destroy(&handle_);
        }
        throw DuckyError(message);
    }

    // Cache column types; logical types are owned and destroyed in the dtor.
    idx_t n_cols = duckdb_appender_column_count(handle_);
    types_.reserve(n_cols);
    type_ids_.reserve(n_cols);
    for (idx_t i = 0; i < n_cols; ++i) {
        duckdb_logical_type lt = duckdb_appender_column_type(handle_, i);
        type_ids_.push_back(duckdb_get_type_id(lt));
        types_.push_back(lt);  // ownership transferred
    }

    // Names + DEFAULT-ness need a separate lookup — the appender API exposes
    // neither. table_description gives both directly.
    try {
        TableIntrospection info = introspect_table(con, catalog, schema, table);
        if (info.names.size() != n_cols) {
            throw DuckyError("ducky: appender for " + quote_ident(table) + " takes " +
                             std::to_string(n_cols) + " columns, but the table describes " +
                             std::to_string(info.names.size()) + " stored columns");
        }
        names_ = std::move(info.names);
        has_default_ = std::move(info.has_default);
    } catch (...) {
        for (auto& lt : types_) duckdb_destroy_logical_type(&lt);
        types_.clear();
        release_gil_if_held release;
        duckdb_appender_destroy(&handle_);
        throw;
    }

    name_to_idx_.reserve(names_.size());
    for (idx_t i = 0; i < (idx_t)names_.size(); ++i) {
        name_to_idx_.emplace(names_[i], i);
    }
}

Appender::~Appender() {
    // Not close(): a destructor must not throw, and the final implicit flush
    // can fail (e.g. a constraint violation). duckdb_appender_destroy closes
    // internally; its error state is dropped here. Call close() (or use a
    // `with` block) to observe flush errors.
    if (handle_) {
        release_gil_if_held release;
        duckdb_appender_destroy(&handle_);
        handle_ = nullptr;
    }
    for (auto& lt : types_) {
        if (lt) duckdb_destroy_logical_type(&lt);
    }
    types_.clear();
}

std::unique_lock<std::mutex> Appender::lock_open() {
    std::unique_lock<std::mutex> lock = lock_releasing_gil(mu_);
    if (!handle_) throw DuckyError("ducky: appender is closed");
    return lock;
}

void Appender::check_state(duckdb_state state, const char* what) {
    if (state != DuckDBError) return;
    std::string message = std::string("ducky: ") + what;
    if (handle_) {
        duckdb_error_data err = duckdb_appender_error_data(handle_);
        if (err && duckdb_error_data_has_error(err)) {
            message += ": ";
            message += duckdb_error_data_message(err);
        }
        duckdb_destroy_error_data(&err);
    }
    throw DuckyError(message);
}

std::vector<std::string> Appender::column_names() const { return names_; }

std::vector<std::string> Appender::column_types() const {
    std::vector<std::string> out;
    out.reserve(type_ids_.size());
    for (auto id : type_ids_) out.emplace_back(duckdb_type_name(id));
    return out;
}

idx_t Appender::resolve(const std::string& name) const {
    auto it = name_to_idx_.find(name);
    if (it == name_to_idx_.end()) {
        throw DuckyError("ducky: unknown column '" + name + "' in appender");
    }
    return it->second;
}

void Appender::flush() {
    auto lock = lock_open();
    duckdb_state state;
    {
        release_gil_if_held release;
        state = duckdb_appender_flush(handle_);
    }
    check_state(state, "appender flush failed");
}

void Appender::close() {
    auto lock = lock_releasing_gil(mu_);
    if (!handle_) return;
    // Best-effort: surface the close error, but always destroy.
    release_gil_if_held release;
    duckdb_state state = duckdb_appender_close(handle_);
    std::string error_message;
    if (state == DuckDBError) {
        duckdb_error_data err = duckdb_appender_error_data(handle_);
        if (err && duckdb_error_data_has_error(err)) {
            error_message = duckdb_error_data_message(err);
        }
        duckdb_destroy_error_data(&err);
    }
    duckdb_appender_destroy(&handle_);
    handle_ = nullptr;
    if (state == DuckDBError) {
        throw DuckyError("ducky: appender close failed: " +
                         (error_message.empty() ? std::string("(no message)") : error_message));
    }
}

namespace {

// One append_row value, converted to its column's exact C type before any of
// the row is handed to DuckDB.
struct Staged {
    enum class Kind : uint8_t {
        Null,
        Default,
        Bool,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        Float,
        Double,
        Varchar,
        Blob,
        Date,
        Time,
        Timestamp,
        Interval,
    };
    Kind kind = Kind::Null;
    union {
        bool b;
        int64_t i;
        uint64_t u;
        double d;
        duckdb_interval interval;
    };
    // Varchar / Blob: borrowed from the Python object, which append_row's
    // arguments keep alive.
    const char* data = nullptr;
    idx_t size = 0;
};

[[noreturn]] void throw_value_error(const std::string& what, duckdb_type t, idx_t col,
                                    const std::string& name) {
    throw DuckyError("ducky: append_row: column " + std::to_string(col) + " ('" + name + "', " +
                     duckdb_type_name(t) + "): " + what);
}

std::string type_name_of(nb::handle value) {
    return nb::cast<std::string>(nb::str(value.type().attr("__name__")));
}

// Integer via __index__ (int, numpy integers), range-checked for the column.
int64_t staged_signed(nb::handle value, int64_t lo, int64_t hi, duckdb_type t, idx_t col,
                      const std::string& name) {
    int overflow = 0;
    long long v = PyLong_AsLongLongAndOverflow(value.ptr(), &overflow);
    if (v == -1 && PyErr_Occurred()) {
        PyErr_Clear();
        throw_value_error("expected an integer, got " + type_name_of(value), t, col, name);
    }
    if (overflow || v < lo || v > hi) {
        throw_value_error("value " + nb::cast<std::string>(nb::str(value)) + " is out of range", t,
                          col, name);
    }
    return (int64_t)v;
}

uint64_t staged_unsigned(nb::handle value, uint64_t hi, duckdb_type t, idx_t col,
                         const std::string& name) {
    nb::object index = nb::steal(PyNumber_Index(value.ptr()));
    if (!index) {
        PyErr_Clear();
        throw_value_error("expected an integer, got " + type_name_of(value), t, col, name);
    }
    unsigned long long v = PyLong_AsUnsignedLongLong(index.ptr());
    if (PyErr_Occurred()) {
        PyErr_Clear();
        throw_value_error("value " + nb::cast<std::string>(nb::str(value)) + " is out of range", t,
                          col, name);
    }
    if (v > hi) {
        throw_value_error("value " + std::to_string(v) + " is out of range", t, col, name);
    }
    return (uint64_t)v;
}

Staged stage_value(duckdb_type t, nb::handle value, idx_t col, const std::string& name) {
    Staged out;
    if (value.is_none()) return out;  // Kind::Null
    PyObject* raw = value.ptr();
    switch (t) {
        case DUCKDB_TYPE_BOOLEAN: {
            // bool and numpy.bool_ (named bool_ before NumPy 2); integers are
            // rejected rather than coerced.
            std::string type = PyBool_Check(raw) ? "bool" : type_name_of(value);
            if (type != "bool" && type != "bool_") {
                throw_value_error("expected bool, got " + type_name_of(value), t, col, name);
            }
            int truth = PyObject_IsTrue(raw);
            if (truth < 0) throw nb::python_error();
            out.kind = Staged::Kind::Bool;
            out.b = truth != 0;
            return out;
        }
        case DUCKDB_TYPE_TINYINT:
            out.kind = Staged::Kind::Int8;
            out.i = staged_signed(value, INT8_MIN, INT8_MAX, t, col, name);
            return out;
        case DUCKDB_TYPE_SMALLINT:
            out.kind = Staged::Kind::Int16;
            out.i = staged_signed(value, INT16_MIN, INT16_MAX, t, col, name);
            return out;
        case DUCKDB_TYPE_INTEGER:
            out.kind = Staged::Kind::Int32;
            out.i = staged_signed(value, INT32_MIN, INT32_MAX, t, col, name);
            return out;
        case DUCKDB_TYPE_BIGINT:
            out.kind = Staged::Kind::Int64;
            out.i = staged_signed(value, INT64_MIN, INT64_MAX, t, col, name);
            return out;
        case DUCKDB_TYPE_UTINYINT:
            out.kind = Staged::Kind::UInt8;
            out.u = staged_unsigned(value, UINT8_MAX, t, col, name);
            return out;
        case DUCKDB_TYPE_USMALLINT:
            out.kind = Staged::Kind::UInt16;
            out.u = staged_unsigned(value, UINT16_MAX, t, col, name);
            return out;
        case DUCKDB_TYPE_UINTEGER:
            out.kind = Staged::Kind::UInt32;
            out.u = staged_unsigned(value, UINT32_MAX, t, col, name);
            return out;
        case DUCKDB_TYPE_UBIGINT:
            out.kind = Staged::Kind::UInt64;
            out.u = staged_unsigned(value, UINT64_MAX, t, col, name);
            return out;
        case DUCKDB_TYPE_FLOAT:
        case DUCKDB_TYPE_DOUBLE: {
            double v = PyFloat_AsDouble(raw);  // float, int, numpy scalars
            if (v == -1.0 && PyErr_Occurred()) {
                PyErr_Clear();
                throw_value_error("expected a number, got " + type_name_of(value), t, col, name);
            }
            out.kind = t == DUCKDB_TYPE_FLOAT ? Staged::Kind::Float : Staged::Kind::Double;
            out.d = v;
            return out;
        }
        case DUCKDB_TYPE_VARCHAR: {
            if (!PyUnicode_Check(raw)) {
                throw_value_error("expected str, got " + type_name_of(value), t, col, name);
            }
            // Borrow PyUnicode's cached UTF-8 buffer; no std::string copy.
            Py_ssize_t size = 0;
            const char* data = PyUnicode_AsUTF8AndSize(raw, &size);
            if (!data) throw nb::python_error();
            out.kind = Staged::Kind::Varchar;
            out.data = data;
            out.size = (idx_t)size;
            return out;
        }
        case DUCKDB_TYPE_BLOB: {
            char* data = nullptr;
            Py_ssize_t size = 0;
            if (!PyBytes_Check(raw) || PyBytes_AsStringAndSize(raw, &data, &size) != 0) {
                PyErr_Clear();
                throw_value_error("expected bytes, got " + type_name_of(value), t, col, name);
            }
            out.kind = Staged::Kind::Blob;
            out.data = data;
            out.size = (idx_t)size;
            return out;
        }
        // Temporal columns take the matching datetime object, mirroring what
        // bind_parameters accepts for `INSERT ... VALUES (?)`.
        case DUCKDB_TYPE_DATE:
            // datetime.datetime is a date subclass; reject it rather than
            // silently truncating the time part.
            if (!PyDate_Check(raw) || PyDateTime_Check(raw)) {
                throw_value_error("expected datetime.date, got " + type_name_of(value), t, col,
                                  name);
            }
            out.kind = Staged::Kind::Date;
            out.i = pydt::date_to_duckdb(raw).days;
            return out;
        case DUCKDB_TYPE_TIME:
            if (!PyTime_Check(raw)) {
                throw_value_error("expected datetime.time, got " + type_name_of(value), t, col,
                                  name);
            }
            out.kind = Staged::Kind::Time;
            out.i = pydt::time_to_duckdb(raw).micros;
            return out;
        case DUCKDB_TYPE_TIMESTAMP:
        case DUCKDB_TYPE_TIMESTAMP_TZ:
            if (!PyDateTime_Check(raw)) {
                throw_value_error("expected datetime.datetime, got " + type_name_of(value), t, col,
                                  name);
            }
            // Aware values land in UTC micros; naive values are taken as-is.
            out.kind = Staged::Kind::Timestamp;
            out.i = pydt::datetime_to_micros(value);
            return out;
        case DUCKDB_TYPE_INTERVAL:
            if (!PyDelta_Check(raw)) {
                throw_value_error("expected datetime.timedelta, got " + type_name_of(value), t, col,
                                  name);
            }
            out.kind = Staged::Kind::Interval;
            out.interval = duckdb_interval{};
            out.interval.days = PyDateTime_DELTA_GET_DAYS(raw);
            out.interval.micros = (int64_t)PyDateTime_DELTA_GET_SECONDS(raw) * 1000000 +
                                  PyDateTime_DELTA_GET_MICROSECONDS(raw);
            return out;
        default:
            throw_value_error(
                "row-API append is not supported for this type; use "
                "append_columns or append_arrow",
                t, col, name);
    }
}

duckdb_state append_staged(duckdb_appender app, const Staged& v) {
    switch (v.kind) {
        case Staged::Kind::Null:
            return duckdb_append_null(app);
        case Staged::Kind::Default:
            return duckdb_append_default(app);
        case Staged::Kind::Bool:
            return duckdb_append_bool(app, v.b);
        case Staged::Kind::Int8:
            return duckdb_append_int8(app, (int8_t)v.i);
        case Staged::Kind::Int16:
            return duckdb_append_int16(app, (int16_t)v.i);
        case Staged::Kind::Int32:
            return duckdb_append_int32(app, (int32_t)v.i);
        case Staged::Kind::Int64:
            return duckdb_append_int64(app, v.i);
        case Staged::Kind::UInt8:
            return duckdb_append_uint8(app, (uint8_t)v.u);
        case Staged::Kind::UInt16:
            return duckdb_append_uint16(app, (uint16_t)v.u);
        case Staged::Kind::UInt32:
            return duckdb_append_uint32(app, (uint32_t)v.u);
        case Staged::Kind::UInt64:
            return duckdb_append_uint64(app, v.u);
        case Staged::Kind::Float:
            return duckdb_append_float(app, (float)v.d);
        case Staged::Kind::Double:
            return duckdb_append_double(app, v.d);
        case Staged::Kind::Varchar:
            return duckdb_append_varchar_length(app, v.data, v.size);
        case Staged::Kind::Blob:
            return duckdb_append_blob(app, v.data, v.size);
        case Staged::Kind::Date:
            return duckdb_append_date(app, duckdb_date{(int32_t)v.i});
        case Staged::Kind::Time:
            return duckdb_append_time(app, duckdb_time{v.i});
        case Staged::Kind::Timestamp:
            return duckdb_append_timestamp(app, duckdb_timestamp{v.i});
        case Staged::Kind::Interval:
            return duckdb_append_interval(app, v.interval);
    }
    return DuckDBError;
}

}  // namespace

void Appender::append_row(nb::args values) {
    auto lock = lock_open();
    idx_t n_cols = type_ids_.size();
    if (values.size() != n_cols) {
        throw DuckyError("ducky: append_row expected " + std::to_string(n_cols) + " values, got " +
                         std::to_string(values.size()));
    }
    // Convert and check every value before appending any: a failure partway
    // through a row would leave DuckDB's appender mid-row, and the only reset
    // (duckdb_appender_clear) also drops every row buffered since the last
    // flush.
    std::vector<Staged> row(n_cols);
    for (idx_t i = 0; i < n_cols; ++i) {
        if (nb::isinstance<DefaultSentinel>(values[i])) {
            if (!has_default_[i]) {
                throw DuckyError("ducky: column '" + names_[i] +
                                 "' has no DEFAULT value; pass an explicit value");
            }
            row[i].kind = Staged::Kind::Default;
            continue;
        }
        row[i] = stage_value(type_ids_[i], values[i], i, names_[i]);
    }
    // Staged values match their columns' types exactly, so these appends don't
    // cast and cannot fail on the values. Appending can flush a full buffer,
    // which takes the client context lock (see the constructor).
    release_gil_if_held release;
    for (idx_t i = 0; i < n_cols; ++i) {
        if (append_staged(handle_, row[i]) == DuckDBError) {
            // Unexpected; reset the half-written row so the appender stays usable.
            duckdb_appender_clear(handle_);
            check_state(DuckDBError, ("append failed at column " + std::to_string(i)).c_str());
        }
    }
    check_state(duckdb_appender_end_row(handle_), "end_row failed");
}

void Appender::append_columns(nb::dict columns, nb::object masks) {
    auto lock = lock_open();
    idx_t n_cols = type_ids_.size();
    if (columns.size() == 0) return;

    std::vector<AnyArray> col_arrays(n_cols);
    std::vector<bool> col_present(n_cols, false);
    std::vector<AnyArray> mask_arrays(n_cols);
    std::vector<bool> mask_present(n_cols, false);

    int64_t n_rows = -1;

    for (auto [key, val] : columns) {
        std::string name = nb::cast<std::string>(key);
        idx_t idx = resolve(name);
        AnyArray arr = nb::cast<AnyArray>(val);
        if (arr.ndim() != 1) {
            throw DuckyError("ducky: column '" + name + "' must be 1-D, got " +
                             std::to_string(arr.ndim()) + "-D");
        }
        FlatDType expect;
        if (!flat_dtype_for(type_ids_[idx], expect)) {
            throw DuckyError("ducky: columnar append for type " +
                             std::string(duckdb_type_name(type_ids_[idx])) +
                             " not supported in v1 (column '" + name + "'); use append_row");
        }
        if (!dtype_eq(arr.dtype(), expect.dtype)) {
            throw DuckyError("ducky: column '" + name + "' dtype mismatch (target is " +
                             duckdb_type_name(type_ids_[idx]) + ")");
        }
        int64_t len = (int64_t)arr.shape(0);
        if (n_rows < 0)
            n_rows = len;
        else if (len != n_rows) {
            throw DuckyError("ducky: column '" + name + "' has length " + std::to_string(len) +
                             ", expected " + std::to_string(n_rows));
        }
        col_arrays[idx] = std::move(arr);
        col_present[idx] = true;
    }

    if (!masks.is_none()) {
        for (auto [key, val] : nb::cast<nb::dict>(masks)) {
            std::string name = nb::cast<std::string>(key);
            idx_t idx = resolve(name);
            if (!col_present[idx]) {
                // A missing column takes its DEFAULT; there is nothing to mask.
                throw DuckyError("ducky: mask for '" + name +
                                 "' has no matching entry in `columns`");
            }
            AnyArray m = nb::cast<AnyArray>(val);
            nb::dlpack::dtype mdt = m.dtype();
            // Check bits too: wider unsigned dtypes share the uint8 type code
            // and would be misread bytewise.
            if (m.ndim() != 1 || mdt.bits != 8 ||
                (mdt.code != nb::dtype<bool>().code && mdt.code != nb::dtype<uint8_t>().code)) {
                throw DuckyError("ducky: mask for '" + name + "' must be 1-D bool/uint8");
            }
            if ((int64_t)m.shape(0) != n_rows) {
                throw DuckyError("ducky: mask for '" + name + "' has wrong length");
            }
            mask_arrays[idx] = std::move(m);
            mask_present[idx] = true;
        }
    }

    if (n_rows <= 0) return;

    std::vector<idx_t> present;
    for (idx_t i = 0; i < n_cols; ++i) {
        if (col_present[i]) present.push_back(i);
    }
    if (present.size() == n_cols) {
        append_column_slices(present, col_arrays, mask_arrays, mask_present, n_rows);
        return;
    }
    // Restrict the appender to the given columns so the others take their
    // DEFAULT (or NULL). Changing the active columns flushes buffered rows.
    duckdb_state state;
    {
        release_gil_if_held release;
        state = duckdb_appender_clear_columns(handle_);
        for (idx_t i : present) {
            if (state == DuckDBError) break;
            state = duckdb_appender_add_column(handle_, names_[i].c_str());
        }
    }
    try {
        check_state(state, "selecting the appended columns failed");
        append_column_slices(present, col_arrays, mask_arrays, mask_present, n_rows);
    } catch (...) {
        release_gil_if_held release;
        duckdb_appender_clear_columns(handle_);
        throw;
    }
    {
        release_gil_if_held release;
        state = duckdb_appender_clear_columns(handle_);
    }
    check_state(state, "flushing the appended columns failed");
}

void Appender::append_column_slices(const std::vector<idx_t>& present,
                                    const std::vector<AnyArray>& arrays,
                                    const std::vector<AnyArray>& masks,
                                    const std::vector<bool>& mask_present, int64_t n_rows) {
    idx_t vec_size = duckdb_vector_size();
    std::vector<duckdb_logical_type> types;
    types.reserve(present.size());
    for (idx_t i : present) types.push_back(types_[i]);

    // Stream slices of up to vec_size rows through one reusable chunk:
    // duckdb_append_data_chunk copies the data into the appender, and
    // duckdb_data_chunk_reset restores capacity and all-valid validity masks
    // (StandardVectorBuffer::ResetCapacity), so per-slice re-allocation is
    // pure overhead.
    duckdb_data_chunk chunk = duckdb_create_data_chunk(types.data(), (idx_t)types.size());
    for (int64_t offset = 0; offset < n_rows; offset += (int64_t)vec_size) {
        idx_t this_n = (idx_t)std::min<int64_t>(vec_size, n_rows - offset);

        duckdb_data_chunk_reset(chunk);
        duckdb_data_chunk_set_size(chunk, this_n);

        for (idx_t k = 0; k < (idx_t)present.size(); ++k) {
            idx_t i = present[k];
            duckdb_vector vec = duckdb_data_chunk_get_vector(chunk, k);
            FlatDType d;
            flat_dtype_for(type_ids_[i], d);
            const uint8_t* src = (const uint8_t*)arrays[i].data();
            std::memcpy(duckdb_vector_get_data(vec), src + offset * d.elem_size,
                        this_n * d.elem_size);

            if (mask_present[i]) {
                duckdb_vector_ensure_validity_writable(vec);
                uint64_t* validity = duckdb_vector_get_validity(vec);
                // bool and uint8 masks are both one byte per row.
                const uint8_t* m = (const uint8_t*)masks[i].data();
                for (idx_t r = 0; r < this_n; ++r) {
                    duckdb_validity_set_row_validity(validity, r, m[offset + r] != 0);
                }
            }
        }

        duckdb_state state;
        {
            // Appending can flush a full buffer (see the constructor).
            release_gil_if_held release;
            state = duckdb_append_data_chunk(handle_, chunk);
        }
        if (state == DuckDBError) {
            duckdb_destroy_data_chunk(&chunk);
            check_state(state, "append_data_chunk failed");
        }
    }
    duckdb_destroy_data_chunk(&chunk);
}

void Appender::append_arrow(nb::object source) {
    auto lock = lock_open();
    duckdb_connection con = connection_->connection;
    if (!con) throw DuckyError("ducky: the appender's connection is closed");

    // Convert one Arrow batch (an ArrowArray matching `converted`) into a data
    // chunk and append it. duckdb_data_chunk_from_arrow takes ownership of the
    // array's buffers, so we null its release afterward to stop the caller's
    // RAII / PyCapsule destructor from double-freeing.
    auto append_batch = [&](duckdb_arrow_converted_schema converted, ArrowArray* array) {
        std::string err;
        duckdb_data_chunk chunk = nullptr;
        if (drain_arrow_error(duckdb_data_chunk_from_arrow(con, array, converted, &chunk), err)) {
            if (chunk) duckdb_destroy_data_chunk(&chunk);
            throw DuckyError("ducky: failed to convert Arrow data to a DuckDB chunk: " + err);
        }
        array->release = nullptr;
        // A batch converts to one chunk of the batch's length, but appending
        // a chunk that needs a cast goes through a vector-size buffer: append
        // in windows of at most duckdb_vector_size() rows.
        duckdb_state st = DuckDBSuccess;
        {
            release_gil_if_held release;
            idx_t rows = duckdb_data_chunk_get_size(chunk);
            idx_t window = duckdb_vector_size();
            if (rows <= window) {
                st = duckdb_append_data_chunk(handle_, chunk);
            } else {
                idx_t n_cols = duckdb_data_chunk_get_column_count(chunk);
                std::vector<duckdb_logical_type> types(n_cols);
                for (idx_t c = 0; c < n_cols; ++c) {
                    types[c] =
                        duckdb_vector_get_column_type(duckdb_data_chunk_get_vector(chunk, c));
                }
                duckdb_data_chunk slice = duckdb_create_data_chunk(types.data(), n_cols);
                for (duckdb_logical_type& t : types) duckdb_destroy_logical_type(&t);
                for (idx_t offset = 0; offset < rows && st == DuckDBSuccess; offset += window) {
                    idx_t count = std::min(window, rows - offset);
                    // Zero-copy windows: a flat vector stays flat.
                    for (idx_t c = 0; c < n_cols; ++c) {
                        duckdb_vector_reference_slice(duckdb_data_chunk_get_vector(slice, c),
                                                      duckdb_data_chunk_get_vector(chunk, c),
                                                      offset, count);
                    }
                    duckdb_data_chunk_set_size(slice, count);
                    st = duckdb_append_data_chunk(handle_, slice);
                }
                duckdb_destroy_data_chunk(&slice);
            }
            duckdb_destroy_data_chunk(&chunk);
        }
        check_state(st, "appending an Arrow batch failed");
    };

    if (nb::hasattr(source, "__arrow_c_stream__")) {
        // Stream path: pyarrow Table / RecordBatchReader / polars / pandas-3 /
        // a ducky Result. The capsule owns the stream and releases it when it
        // goes out of scope here; the schema and per-batch arrays it hands us,
        // we own (hence the *Release guards below).
        nb::object capsule = source.attr("__arrow_c_stream__")();
        auto* stream = (ArrowArrayStream*)PyCapsule_GetPointer(capsule.ptr(), "arrow_array_stream");
        if (!stream) {
            PyErr_Clear();
            throw DuckyError(
                "ducky: __arrow_c_stream__ did not return an 'arrow_array_stream' capsule");
        }

        ArrowSchema schema{};
        if (stream->get_schema(stream, &schema) != 0) {
            const char* m = stream->get_last_error ? stream->get_last_error(stream) : nullptr;
            throw DuckyError(std::string("ducky: Arrow stream get_schema failed") +
                             (m ? std::string(": ") + m : ""));
        }
        SchemaRelease schema_guard{&schema};

        std::string err;
        ConvertedSchemaRelease conv;
        if (drain_arrow_error(duckdb_schema_from_arrow(con, &schema, &conv.schema), err)) {
            throw DuckyError("ducky: failed to convert the Arrow schema: " + err);
        }

        for (;;) {
            ArrowArray array{};
            if (stream->get_next(stream, &array) != 0) {
                const char* m = stream->get_last_error ? stream->get_last_error(stream) : nullptr;
                throw DuckyError(std::string("ducky: Arrow stream get_next failed") +
                                 (m ? std::string(": ") + m : ""));
            }
            if (!array.release) break;  // end of stream
            ArrayRelease array_guard{&array};
            append_batch(conv.schema, &array);
        }
    } else if (nb::hasattr(source, "__arrow_c_array__")) {
        // Single-batch path: a pyarrow RecordBatch. The (schema, array) capsules
        // own their structs and release them when `pair` drops at scope exit, so
        // we don't add our own guards — append_batch only nulls the array's
        // release after consuming it, which the array capsule honors.
        nb::tuple pair = nb::cast<nb::tuple>(source.attr("__arrow_c_array__")());
        auto* schema = (ArrowSchema*)PyCapsule_GetPointer(pair[0].ptr(), "arrow_schema");
        auto* array = (ArrowArray*)PyCapsule_GetPointer(pair[1].ptr(), "arrow_array");
        if (!schema || !array) {
            PyErr_Clear();
            throw DuckyError("ducky: __arrow_c_array__ did not return (schema, array) capsules");
        }
        std::string err;
        ConvertedSchemaRelease conv;
        if (drain_arrow_error(duckdb_schema_from_arrow(con, schema, &conv.schema), err)) {
            throw DuckyError("ducky: failed to convert the Arrow schema: " + err);
        }
        append_batch(conv.schema, array);
    } else {
        throw DuckyError(
            "ducky: append_arrow() needs an object implementing the Arrow PyCapsule interface "
            "(__arrow_c_stream__ or __arrow_c_array__)");
    }
}
