#include "ducky.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <atomic>

#include "aggregate.hpp"
#include "appender.hpp"
#include "chunk.hpp"
#include "connection.hpp"
#include "duckdb.h"
#include "function.hpp"
#include "prepared.hpp"
#include "result.hpp"
#include "table.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

// Cached handle to the ducky._conversions module, imported once on first use.
// Avoids re-acquiring the import lock + redoing the sys.modules lookup on every
// .arrow() / .to_numpy() / .chunks() call. See cached_singleton() in ducky.hpp
// for the leak + GIL-release rationale.
const nb::module_& conversions() {
    static std::atomic<nb::module_*> cached{nullptr};
    static nb::ft_mutex mu;
    return cached_singleton(cached, mu, [] { return nb::module_::import_("ducky._conversions"); });
}

// Register the DataFrame/array/Arrow accessors that both Result and Connection
// expose. Every method just forwards to the matching helper in
// ducky._conversions; `src` maps the bound object to the Arrow/chunk source the
// helper consumes — identity for Result, `.current_result()` for Connection.
// Defining them here once (instead of per class) keeps the two surfaces in
// lockstep; they stay compiled methods so nanobind's stub generator still emits
// them into _core.pyi.
template <typename Cls, typename Src>
void def_conversions(Cls& cls, Src src) {
    cls.def(
        "arrow", [src](nb::object self) { return conversions().attr("arrow")(src(self)); },
        nb::sig("def arrow(self) -> pyarrow.Table"), "Return the result as a pyarrow.Table.");
    cls.def(
        "df", [src](nb::object self) { return conversions().attr("df")(src(self)); },
        nb::sig("def df(self) -> pandas.DataFrame"), "Return the result as a pandas.DataFrame.");
    cls.def(
        "pl",
        [src](nb::object self, bool lazy) { return conversions().attr("pl")(src(self), lazy); },
        "lazy"_a = false,
        nb::sig("def pl(self, lazy: bool = False) -> polars.DataFrame | polars.LazyFrame"),
        "Return the result as a polars DataFrame (or LazyFrame).");
    cls.def(
        "fetchnumpy",
        [src](nb::object self) { return conversions().attr("fetchnumpy")(src(self)); },
        nb::sig("def fetchnumpy(self) -> dict[str, numpy.ndarray]"),
        "Return the result as a dict of column name -> numpy array.");
    cls.def(
        "chunks", [src](nb::object self) { return conversions().attr("chunks")(src(self)); },
        nb::sig("def chunks(self) -> collections.abc.Iterator[Chunk]"),
        "Iterate over the result one Chunk at a time. Drains the result.");
    cls.def(
        "iter_batches",
        [src](nb::object self, nb::object columns, bool with_validity) {
            return conversions().attr("iter_batches")(src(self), columns, with_validity);
        },
        "columns"_a = nb::none(), "with_validity"_a = false,
        nb::sig("def iter_batches(self, columns: collections.abc.Iterable[str] | None = "
                "None, with_validity: bool = False) -> collections.abc.Iterator["
                "dict[str, numpy.ndarray | tuple[numpy.ndarray, numpy.ndarray | None]]]"),
        "Yield one dict per chunk: {name: ndarray}, or {name: (values, mask)} "
        "if with_validity=True.");
    cls.def(
        "iter_batches_torch",
        [src](nb::object self, nb::object columns, nb::object device) {
            return conversions().attr("iter_batches_torch")(src(self), columns, device);
        },
        "columns"_a = nb::none(), "device"_a = nb::none(),
        nb::sig("def iter_batches_torch(self, columns: collections.abc.Iterable[str] | None = "
                "None, device: torch.device | str | int | None = None) "
                "-> collections.abc.Iterator[dict[str, torch.Tensor]]"),
        "Yield one dict per chunk: {name: torch.Tensor}. Zero-copy on CPU via DLPack.");
    cls.def(
        "iter_batches_jax",
        [src](nb::object self, nb::object columns, nb::object device) {
            return conversions().attr("iter_batches_jax")(src(self), columns, device);
        },
        "columns"_a = nb::none(), "device"_a = nb::none(),
        nb::sig("def iter_batches_jax(self, columns: collections.abc.Iterable[str] | None = "
                "None, device: jax.Device | None = None) "
                "-> collections.abc.Iterator[dict[str, jax.Array]]"),
        "Yield one dict per chunk: {name: jax.Array}.");
    cls.def(
        "to_numpy",
        [src](nb::object self, nb::object columns) {
            return conversions().attr("to_numpy")(src(self), columns);
        },
        "columns"_a = nb::none(),
        nb::sig("def to_numpy(self, columns: collections.abc.Iterable[str] | None = None) "
                "-> dict[str, numpy.ndarray]"),
        "Eagerly concatenate all chunks into {name: numpy.ndarray}.");
    cls.def(
        "to_torch",
        [src](nb::object self, nb::object columns, nb::object device) {
            return conversions().attr("to_torch")(src(self), columns, device);
        },
        "columns"_a = nb::none(), "device"_a = nb::none(),
        nb::sig("def to_torch(self, columns: collections.abc.Iterable[str] | None = None, "
                "device: torch.device | str | int | None = None) "
                "-> dict[str, torch.Tensor]"),
        "Eagerly concatenate all chunks into {name: torch.Tensor}.");
    cls.def(
        "to_jax",
        [src](nb::object self, nb::object columns, nb::object device) {
            return conversions().attr("to_jax")(src(self), columns, device);
        },
        "columns"_a = nb::none(), "device"_a = nb::none(),
        nb::sig("def to_jax(self, columns: collections.abc.Iterable[str] | None = None, "
                "device: jax.Device | None = None) -> dict[str, jax.Array]"),
        "Eagerly concatenate all chunks into {name: jax.Array}.");
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "ducky: tiny nanobind bindings for the DuckDB C API";
    m.attr("__duckdb_version__") = duckdb_library_version();

    // C++ DuckyError -> Python ducky.Error.
    // NOLINTNEXTLINE(bugprone-unused-raii)
    nb::exception<DuckyError>(m, "Error");

    nb::class_<Chunk>(m, "Chunk",
                      "A single DuckDB data chunk: a column-major batch of up "
                      "to STANDARD_VECTOR_SIZE rows. Numeric and temporal "
                      "columns are exposed as zero-copy ndarrays.")
        .def("__len__", &Chunk::size, "Number of rows in this chunk.")
        .def_prop_ro("columns", &Chunk::column_names, "Column names.")
        .def_prop_ro("types", &Chunk::column_types, "Column type names.")
        .def(
            "column",
            [](nb::object self, nb::object key) {
                return nb::cast<Chunk&>(self).column(key, self);
            },
            "key"_a, nb::sig("def column(self, key: int | str) -> numpy.ndarray"),
            "Return the column at `key` (int index or str name) as a 1-D "
            "ndarray view over the chunk's buffer.")
        .def(
            "validity",
            [](nb::object self, nb::object key) {
                return nb::cast<Chunk&>(self).validity(key, self);
            },
            "key"_a, nb::sig("def validity(self, key: int | str) -> numpy.ndarray | None"),
            "Return a uint8 ndarray (1=valid, 0=null) of length len(self), or "
            "None if the column has no nulls.")
        .def("decimal_scale", &Chunk::decimal_scale, "key"_a,
             nb::sig("def decimal_scale(self, key: int | str) -> int"),
             "Return the scale (digits after the decimal point) for a DECIMAL "
             "column. Raises Error if the column is not DECIMAL.")
        .def(
            "dlpack",
            [](nb::object self, nb::object key) {
                return nb::cast<Chunk&>(self).dlpack(key, self);
            },
            "key"_a, nb::sig("def dlpack(self, key: int | str) -> typing.Any"),
            "Return the column at `key` as an object implementing the DLPack "
            "protocol (__dlpack__ / __dlpack_device__), without going through "
            "numpy. Flat numeric/temporal types only.");

    nb::class_<Result> result_cls(m, "Result",
                                  "A query result. Iterate it, or use the fetch* methods, "
                                  "to pull rows as tuples.");
    result_cls.def_prop_ro("columns", &Result::column_names, "Column names.")
        .def_prop_ro("types", &Result::column_types, "Column type names.")
        .def_prop_ro("description", &Result::description,
                     nb::sig("def description(self) -> list[tuple] | None"),
                     "PEP 249 result description.")
        .def("fetchone", &Result::fetchone, nb::sig("def fetchone(self) -> tuple | None"),
             "Return the next row, or None.")
        .def("fetchmany", &Result::fetchmany, "size"_a = 1,
             nb::sig("def fetchmany(self, size: int = 1) -> list[tuple]"),
             "Return up to `size` rows.")
        .def("fetchall", &Result::fetchall, nb::sig("def fetchall(self) -> list[tuple]"),
             "Return all remaining rows.")
        .def("fetch_chunk", &Result::fetch_chunk, nb::sig("def fetch_chunk(self) -> Chunk | None"),
             "Pull the next data chunk as a Chunk, or None at end of stream.")
        .def(
            "__arrow_c_stream__",
            [](nb::object self, nb::object) {
                return nb::cast<Result&>(self).arrow_c_stream(self);
            },
            "requested_schema"_a = nb::none(),
            nb::sig("def __arrow_c_stream__(self, requested_schema: typing.Any = None) "
                    "-> typing.Any"),
            "Export the result via the Arrow C stream (PyCapsule) interface.");
    // The Result *is* the Arrow/chunk source the conversion helpers consume.
    def_conversions(result_cls, [](nb::object self) { return self; });
    result_cls
        .def(
            "__iter__", [](nb::object self) { return self; },
            nb::sig("def __iter__(self) -> collections.abc.Iterator[tuple]"))
        .def(
            "__next__",
            [](Result& self) {
                nb::object row = self.fetchone();
                if (row.is_none()) throw nb::stop_iteration();
                return row;
            },
            nb::sig("def __next__(self) -> tuple"));

    nb::class_<Appender>(m, "Appender",
                         "Bulk-insert handle for a target table. Supports a "
                         "row-at-a-time API for mixed/typed writes and a "
                         "columnar fast path for numeric/temporal ndarrays.")
        .def_prop_ro("columns", &Appender::column_names, "Target column names.")
        .def_prop_ro("types", &Appender::column_types, "Target column type names.")
        .def("append_row", &Appender::append_row,
             nb::sig("def append_row(self, *values: object) -> None"),
             "Append one row of positional values, matching the target column order.")
        .def("append_columns", &Appender::append_columns, "columns"_a, "masks"_a = nb::none(),
             nb::sig("def append_columns(self, columns: dict[str, numpy.ndarray], "
                     "masks: dict[str, numpy.ndarray] | None = None) -> None"),
             "Append a batch of columns. Missing columns are filled with NULL. "
             "`masks` maps column name -> 1-D bool/uint8 array (True = valid).")
        .def("flush", &Appender::flush, "Flush pending rows to the table.")
        .def("close", &Appender::close, "Flush and tear down the appender.")
        .def(
            "__enter__", [](Appender& self) -> Appender& { return self; }, nb::rv_policy::reference)
        .def(
            "__exit__", [](Appender& self, nb::object, nb::object, nb::object) { self.close(); },
            "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none(),
            nb::sig("def __exit__(self, exc_type: type[BaseException] | None, exc_value: "
                    "BaseException | None, traceback: types.TracebackType | None) -> None"));

    nb::class_<PreparedStatement>(
        m, "PreparedStatement",
        "A SQL statement compiled once via Connection.prepare(), executable "
        "repeatedly with different parameters without re-parsing the query.")
        .def("execute", &PreparedStatement::execute, "parameters"_a = nb::none(),
             nb::sig("def execute(self, parameters: list | tuple | dict[str, typing.Any] | None = "
                     "None) -> Result"),
             "Bind `parameters` (positional list/tuple, named dict, or None) and "
             "run the statement, returning its Result.")
        .def("executemany", &PreparedStatement::executemany, "parameters"_a,
             nb::sig("def executemany(self, parameters: collections.abc.Iterable[list | tuple | "
                     "dict[str, typing.Any]]) -> None"),
             "Run the statement once per parameter set, discarding each result. "
             "The fast path for batched INSERT/UPDATE/DELETE.")
        .def_prop_ro("num_parameters", &PreparedStatement::num_parameters,
                     "Number of bind parameters in the statement.")
        .def("parameter_name", &PreparedStatement::parameter_name, "index"_a,
             nb::sig("def parameter_name(self, index: int) -> str | None"),
             "Name of the parameter at `index` (1-based), or None if the index is "
             "out of range or the parameter is positional.")
        .def_prop_ro("columns", &PreparedStatement::column_names,
                     "Result column names, known ahead of execution (empty for "
                     "statements that produce no result set).")
        .def_prop_ro("types", &PreparedStatement::column_types,
                     "Result column type names, known ahead of execution.")
        .def_prop_ro("statement_type", &PreparedStatement::statement_type,
                     nb::sig("def statement_type(self) -> str"),
                     "The statement kind: 'SELECT', 'INSERT', 'UPDATE', etc.")
        .def("close", &PreparedStatement::close, "Destroy the prepared statement.")
        .def(
            "__enter__", [](PreparedStatement& self) -> PreparedStatement& { return self; },
            nb::rv_policy::reference)
        .def(
            "__exit__",
            [](PreparedStatement& self, nb::object, nb::object, nb::object) { self.close(); },
            "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none(),
            nb::sig("def __exit__(self, exc_type: type[BaseException] | None, exc_value: "
                    "BaseException | None, traceback: types.TracebackType | None) -> None"));

    nb::class_<Connection> conn_cls(m, "Connection", "A connection to a DuckDB database.");
    conn_cls
        .def("execute", &Connection::execute, "query"_a, "parameters"_a = nb::none(),
             nb::rv_policy::reference,
             nb::sig("def execute(self, query: str, parameters: list | tuple | "
                     "dict[str, typing.Any] | None = None) -> Connection"),
             "Execute a query, optionally with positional parameters, and "
             "return self for chaining.")
        // The returned Result shares the DuckDBHandle, so it keeps the database
        // open on its own — no keep_alive needed.
        .def("sql", &Connection::sql, "query"_a, "Run a query and return its Result.")
        .def("query", &Connection::sql, "query"_a, "Alias for sql().")
        .def("prepare", &Connection::prepare, "query"_a,
             nb::sig("def prepare(self, query: str) -> PreparedStatement"),
             "Compile `query` once into a PreparedStatement for repeated execution "
             "with different parameters, avoiding per-call re-parsing and planning.")
        .def("fetchone", &Connection::fetchone, nb::sig("def fetchone(self) -> tuple | None"))
        .def("fetchmany", &Connection::fetchmany, "size"_a = 1,
             nb::sig("def fetchmany(self, size: int = 1) -> list[tuple]"))
        .def("fetchall", &Connection::fetchall, nb::sig("def fetchall(self) -> list[tuple]"))
        .def_prop_ro("description", &Connection::description,
                     nb::sig("def description(self) -> list[tuple] | None"))
        .def_prop_ro("columns", &Connection::columns,
                     nb::sig("def columns(self) -> list[str] | None"))
        .def_prop_ro("current_result", &Connection::current_result,
                     nb::sig("def current_result(self) -> Result"),
                     "The most recent result produced by execute(); raises Error if no "
                     "query has run yet. The conversion accessors (arrow/df/pl/...) "
                     "delegate to it.");
    // Connection's conversion accessors operate on its most recent result; the
    // Result itself is the Arrow/chunk source the helpers consume.
    def_conversions(conn_cls, [](nb::object self) {
        return nb::cast(nb::cast<Connection&>(self).current_result());
    });
    conn_cls
        .def(
            "create_function",
            [](Connection& self, const std::string& name, nb::callable fn, nb::object parameters,
               nb::object return_type, nb::object varargs) {
                create_scalar_function(self, name, std::move(fn), parameters, return_type, varargs);
            },
            "name"_a, "fn"_a, "parameters"_a = nb::none(), "return_type"_a = nb::none(),
            "varargs"_a = nb::none(),
            nb::sig("def create_function(self, name: str, fn: collections.abc.Callable, "
                    "parameters: list[str] | dict[str, str] | None = None, "
                    "return_type: str | None = None, "
                    "varargs: str | None = None) -> None"),
            "Register a Python callable as a DuckDB scalar function. "
            "`parameters` is a list of type strings (positional call) or a "
            "dict of {name: type_string} (dict-style call). Inputs arrive as "
            "zero-copy 1-D ndarrays; `fn` must return one ndarray of length "
            "chunk_size and matching dtype. If `parameters` or `return_type` is "
            "omitted, they are inferred from `fn`'s annotations (bool/int/float "
            "→ BOOLEAN/BIGINT/DOUBLE). Pass `varargs=\"TYPE\"` (mutually exclusive "
            "with `parameters`) to register a variable-arity function; `fn` is "
            "then called as `fn(*args)` with one ndarray per SQL argument.")
        .def(
            "create_arrow_function",
            [](Connection& self, const std::string& name, nb::callable fn, nb::object parameters,
               const std::string& return_type, bool record_batch) {
                create_arrow_scalar_function(self, name, std::move(fn), parameters, return_type,
                                             record_batch);
            },
            "name"_a, "fn"_a, "parameters"_a, "return_type"_a, "record_batch"_a = false,
            nb::sig("def create_arrow_function(self, name: str, fn: collections.abc.Callable, "
                    "parameters: list[str] | dict[str, str], return_type: str, "
                    "record_batch: bool = False) -> None"),
            "Register a Python callable as a DuckDB scalar function backed by the "
            "Arrow C-API path; supports VARCHAR, LIST, STRUCT, DECIMAL, MAP, and "
            "any other DuckDB type. `parameters` is a list of DuckDB type strings "
            "or a dict of {name: type_string}. By default `fn` is called with one "
            "`pyarrow.Array` per column (positional, or a {name: Array} dict when "
            "`parameters` is a dict). Pass `record_batch=True` to receive a single "
            "`pyarrow.RecordBatch` instead. `fn` must return a `pyarrow.Array`.")
        .def(
            "create_aggregate_function",
            [](Connection& self, const std::string& name, nb::object cls, nb::object parameters,
               const std::string& return_type) {
                create_aggregate_function(self, name, cls, parameters, return_type);
            },
            "name"_a, "cls"_a, "parameters"_a, "return_type"_a,
            nb::sig("def create_aggregate_function(self, name: str, cls: type, "
                    "parameters: list[str] | dict[str, str], return_type: str) -> None"),
            "Register a Python class as a DuckDB aggregate function. `cls` must "
            "have `__init__`, `update(self, *arrays)`, and `finalize(self) -> scalar` "
            "methods. An optional `combine(self, other)` method enables parallel "
            "aggregate execution.")
        .def(
            "create_table_function",
            [](Connection& self, const std::string& name, nb::callable factory,
               nb::object parameters, nb::object columns) {
                create_table_function(self, name, factory, parameters, columns);
            },
            "name"_a, "factory"_a, "parameters"_a, "columns"_a,
            nb::sig("def create_table_function(self, name: str, "
                    "factory: collections.abc.Callable, "
                    "parameters: list[str], columns: dict[str, str]) -> None"),
            "Register a Python generator function as a DuckDB table function. "
            "`factory` is called with the SQL arguments and must return a generator "
            "that yields one tuple per row. `columns` is an ordered dict "
            "{column_name: type_string} declaring the output schema.")
        .def(
            "appender",
            [](Connection& self, const std::string& table, std::optional<std::string> schema,
               std::optional<std::string> catalog) {
                return new Appender(self, table, std::move(schema), std::move(catalog));
            },
            "table"_a, "schema"_a = nb::none(), "catalog"_a = nb::none(),
            nb::sig("def appender(self, table: str, schema: str | None = None, "
                    "catalog: str | None = None) -> Appender"),
            "Open an Appender for bulk inserts into `table`. The appender shares "
            "the connection's database handle, so the database stays open until "
            "the appender is closed.")
        .def("interrupt", &Connection::interrupt,
             "Best-effort cancel of any query currently running on this connection. "
             "Safe to call from another thread while execute() is in flight.")
        .def("progress", &Connection::progress,
             nb::sig("def progress(self) -> tuple[float, int, int]"),
             "Snapshot of the current query's progress: "
             "(percentage, rows_processed, total_rows_to_process). "
             "`percentage` is -1.0 until DuckDB has an estimate.")
        .def("register_arrow", &Connection::register_arrow, "name"_a, "obj"_a,
             nb::sig("def register_arrow(self, name: str, obj: typing.Any) -> None"),
             "Register a Python object exposing `__arrow_c_stream__` (pyarrow "
             "Table, polars DataFrame, pandas-3 DataFrame, ...) as a table named "
             "`name`. The data is materialized into DuckDB at registration; "
             "the source object is not retained.")
        .def("close", &Connection::close, "Close the connection.")
        .def(
            "__enter__", [](Connection& self) -> Connection& { return self; },
            nb::rv_policy::reference)
        .def(
            "__exit__", [](Connection& self, nb::object, nb::object, nb::object) { self.close(); },
            "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none(),
            nb::sig("def __exit__(self, exc_type: type[BaseException] | None, exc_value: "
                    "BaseException | None, traceback: types.TracebackType | None) -> None"));

    m.def(
        "connect",
        [](const std::string& database, nb::object config) {
            return new Connection(database, std::move(config));
        },
        "database"_a = ":memory:", "config"_a = nb::none(),
        nb::sig("def connect(database: str = ':memory:', config: dict[str, str] | None = None) "
                "-> Connection"),
        "Open `database` (default in-memory) and return a Connection. "
        "`config` is an optional dict of DuckDB settings applied at open time "
        "(see ducky.config_options() for the full list of keys).");

    m.def(
        "config_options",
        []() {
            size_t n = duckdb_config_count();
            nb::list out;
            for (size_t i = 0; i < n; ++i) {
                const char* name = nullptr;
                const char* desc = nullptr;
                if (duckdb_get_config_flag(i, &name, &desc) == DuckDBSuccess) {
                    out.append(nb::make_tuple(name ? name : "", desc ? desc : ""));
                }
            }
            return out;
        },
        nb::sig("def config_options() -> list[tuple[str, str]]"),
        "Return the (name, description) of every DuckDB config option "
        "settable via connect(config=...).");
}
