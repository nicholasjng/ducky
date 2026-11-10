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

// Cached module handles.
const nb::module_& conversions() {
    static std::atomic<nb::module_*> cached{nullptr};
    static nb::ft_mutex mu;
    return cached_singleton(cached, mu, [] { return nb::module_::import_("ducky._conversions"); });
}

const nb::module_& aio() {
    static std::atomic<nb::module_*> cached{nullptr};
    static nb::ft_mutex mu;
    return cached_singleton(cached, mu, [] { return nb::module_::import_("ducky._aio"); });
}

// Register conversions shared by Result and Connection.
template <typename Cls, typename Src>
void def_conversions(Cls& cls, Src src) {
    cls.def(
        "arrow", [src](nb::object self) { return conversions().attr("arrow")(src(self)); },
        nb::sig("def arrow(self) -> pyarrow.Table"), "Return the result as a pyarrow.Table.");
    cls.def(
        "arrow_batches",
        [src](nb::object self) { return conversions().attr("arrow_batches")(src(self)); },
        nb::sig("def arrow_batches(self) -> pyarrow.RecordBatchReader"),
        "Stream one Arrow record batch per DuckDB chunk.");
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
        "Return the result as a dict of column name -> numpy array. Numeric and "
        "temporal columns only; use arrow() for strings and nested types. NULLs become "
        "NaN in FLOAT/DOUBLE columns and raise elsewhere.");
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
        "Yield Torch tensors per chunk.");
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
        "iter_batches_mlx",
        [src](nb::object self, nb::object columns) {
            return conversions().attr("iter_batches_mlx")(src(self), columns);
        },
        "columns"_a = nb::none(),
        nb::sig("def iter_batches_mlx(self, columns: collections.abc.Iterable[str] | None = "
                "None) -> collections.abc.Iterator[dict[str, mlx.core.array]]"),
        "Yield MLX arrays per chunk.");
    cls.def(
        "to_torch",
        [src](nb::object self, nb::object columns, nb::object device) {
            return conversions().attr("to_torch")(src(self), columns, device);
        },
        "columns"_a = nb::none(), "device"_a = nb::none(),
        nb::sig("def to_torch(self, columns: collections.abc.Iterable[str] | None = None, "
                "device: torch.device | str | int | None = None) "
                "-> dict[str, torch.Tensor]"),
        "Return selected columns as Torch tensors.");
    cls.def(
        "to_jax",
        [src](nb::object self, nb::object columns, nb::object device) {
            return conversions().attr("to_jax")(src(self), columns, device);
        },
        "columns"_a = nb::none(), "device"_a = nb::none(),
        nb::sig("def to_jax(self, columns: collections.abc.Iterable[str] | None = None, "
                "device: jax.Device | None = None) -> dict[str, jax.Array]"),
        "Return selected columns as JAX arrays.");
    cls.def(
        "to_mlx",
        [src](nb::object self, nb::object columns) {
            return conversions().attr("to_mlx")(src(self), columns);
        },
        "columns"_a = nb::none(),
        nb::sig("def to_mlx(self, columns: collections.abc.Iterable[str] | None = None) "
                "-> dict[str, mlx.core.array]"),
        "Return selected columns as MLX arrays.");
}

// Let Python collect cycles through sinks and registered Arrow sources.
int connection_tp_traverse(PyObject* self, visitproc visit, void* arg) {
    Py_VISIT(Py_TYPE(self));
    if (!nb::inst_ready(self)) return 0;
    return nb::inst_ptr<Connection>(self)->gc_traverse(visit, arg);
}

int connection_tp_clear(PyObject* self) {
    if (nb::inst_ready(self)) nb::inst_ptr<Connection>(self)->gc_clear();
    return 0;
}

PyType_Slot connection_slots[] = {
    {Py_tp_traverse, (void*)connection_tp_traverse},
    {Py_tp_clear, (void*)connection_tp_clear},
    {0, nullptr},
};

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "ducky: tiny nanobind bindings for the DuckDB C API";
    m.attr("__duckdb_version__") = duckdb_library_version();

    // NOLINTNEXTLINE(bugprone-unused-raii)
    nb::exception<DuckyError>(m, "Error");

    nb::class_<DefaultSentinel>(m, "Default",
                                "Sentinel type for appender DEFAULT values; use ``ducky.DEFAULT``.")
        .def("__repr__", [](const DefaultSentinel&) { return "ducky.DEFAULT"; });
    m.attr("DEFAULT") = nb::cast(DefaultSentinel{});

    nb::class_<Chunk>(m, "Chunk", nb::pooled(), "A column-major DuckDB result batch.")
        .def("__len__", &Chunk::size, "Number of rows.")
        .def_prop_ro("columns", &Chunk::column_names, "Column names.")
        .def_prop_ro("types", &Chunk::column_types, "Column type names.")
        .def(
            "column",
            [](nb::object self, nb::object key) {
                return nb::cast<Chunk&>(self).column(key, self);
            },
            "key"_a, nb::sig("def column(self, key: int | str) -> numpy.ndarray"),
            "Return a zero-copy 1-D ndarray view by index or name.")
        .def(
            "validity",
            [](nb::object self, nb::object key) {
                return nb::cast<Chunk&>(self).validity(key, self);
            },
            "key"_a, nb::sig("def validity(self, key: int | str) -> numpy.ndarray | None"),
            "Return a uint8 validity mask (1 = valid, 0 = NULL), or None if no nulls.")
        .def("decimal_scale", &Chunk::decimal_scale, "key"_a,
             nb::sig("def decimal_scale(self, key: int | str) -> int"),
             "Return a DECIMAL column's scale.")
        .def(
            "dlpack",
            [](nb::object self, nb::object key) {
                return nb::cast<Chunk&>(self).dlpack(key, self);
            },
            "key"_a, nb::sig("def dlpack(self, key: int | str) -> typing.Any"),
            "Return a numeric or temporal column as a DLPack object.")
        .def(
            "mlx",
            [](nb::object self, nb::object key) { return nb::cast<Chunk&>(self).mlx(key, self); },
            "key"_a, nb::sig("def mlx(self, key: int | str) -> mlx.core.array"),
            "Return a numeric or temporal column as an MLX array.");

    nb::class_<Result> result_cls(m, "Result",
                                  "A query result. Iterate or call ``fetch*`` to pull rows.");
    result_cls.def_prop_ro("columns", &Result::column_names, "Column names.")
        .def_prop_ro("types", &Result::column_types, "Column type names.")
        .def_prop_ro("description", &Result::description,
                     nb::sig("def description(self) -> list[tuple] | None"),
                     "PEP 249 result description, or None if the result is closed.")
        .def("fetchone", &Result::fetchone, nb::sig("def fetchone(self) -> tuple | None"),
             "Return the next row as a tuple, or None at end of result.", nb::lock_self())
        .def("fetchmany", &Result::fetchmany, "size"_a = 1,
             nb::sig("def fetchmany(self, size: int = 1) -> list[tuple]"),
             "Return up to ``size`` rows.", nb::lock_self())
        .def("fetchall", &Result::fetchall, nb::sig("def fetchall(self) -> list[tuple]"),
             "Return all remaining rows.", nb::lock_self())
        .def("fetchitem", &Result::fetchitem, nb::sig("def fetchitem(self) -> typing.Any"),
             "Return the lone scalar of a 1x1 result, for ``COUNT(*)``-style queries. "
             "Raises :class:`Error` on any other shape.",
             nb::lock_self())
        .def_prop_ro("rows_changed", &Result::rows_changed,
                     "Rows affected by INSERT, UPDATE, or DELETE; otherwise 0.", nb::lock_self())
        .def("fetch_chunk", &Result::fetch_chunk, nb::sig("def fetch_chunk(self) -> Chunk | None"),
             "Pull the next :class:`Chunk`, or None at end of stream.", nb::lock_self())
        .def("fetch_numpy_chunk", &Result::fetch_numpy_chunk,
             nb::sig("def fetch_numpy_chunk(self) -> dict[str, numpy.ndarray] | None"),
             "Pull the next chunk as NumPy arrays, or None at end.", nb::lock_self())
        .def("to_numpy", &Result::to_numpy, "columns"_a = nb::none(),
             nb::sig("def to_numpy(self, columns: collections.abc.Iterable[str] "
                     "| None = None) -> dict[str, numpy.ndarray]"),
             "Drain the result into a {name: numpy.ndarray} dict. NULLs become NaN in "
             "FLOAT/DOUBLE columns and raise elsewhere. Raises on VARCHAR/LIST/STRUCT/MAP; "
             "use arrow() for those.",
             nb::lock_self())
        .def(
            "__arrow_c_stream__",
            [](nb::object self, nb::object) {
                return nb::cast<Result&>(self).arrow_c_stream(self);
            },
            "requested_schema"_a = nb::none(),
            nb::sig("def __arrow_c_stream__(self, requested_schema: typing.Any = None) "
                    "-> typing.Any"),
            "Export the result via the Arrow PyCapsule stream interface.", nb::lock_self())
        .def(
            "_arrow_chunk_stream",
            [](nb::object self) {
                return nb::cast<Result&>(self).arrow_c_stream(self, /*allow_fast=*/false);
            },
            nb::sig("def _arrow_chunk_stream(self) -> typing.Any"),
            "Return a per-chunk Arrow C stream capsule.", nb::lock_self());
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
            nb::sig("def __next__(self) -> tuple"), nb::lock_self());

    nb::class_<Appender>(m, "Appender", "Buffered bulk-insert handle for a table.")
        .def_prop_ro("columns", &Appender::column_names, "Target column names.")
        .def_prop_ro("types", &Appender::column_types, "Target column type names.")
        .def("append_row", &Appender::append_row,
             nb::sig("def append_row(self, *values: object) -> None"),
             "Append one row of positional values, in target column order.", nb::lock_self())
        .def("append_columns", &Appender::append_columns, "columns"_a, "masks"_a = nb::none(),
             nb::sig("def append_columns(self, columns: dict[str, numpy.ndarray], "
                     "masks: dict[str, numpy.ndarray] | None = None) -> None"),
             "Append equal-length numeric or temporal arrays with optional validity masks. "
             "Missing columns take their DEFAULT, or NULL.",
             nb::lock_self())
        .def("append_arrow", &Appender::append_arrow, "source"_a,
             nb::sig("def append_arrow(self, source: typing.Any) -> None"),
             "Append an Arrow stream or array. Columns must match the table by position "
             "and type.",
             nb::lock_self())
        .def("flush", &Appender::flush, "Flush pending rows to the table.", nb::lock_self())
        .def("close", &Appender::close, "Flush and tear down the appender.", nb::lock_self())
        .def(
            "__enter__", [](Appender& self) -> Appender& { return self; }, nb::rv_policy::reference,
            nb::sig("def __enter__(self) -> typing.Self"))
        .def(
            "__exit__", [](Appender& self, nb::object, nb::object, nb::object) { self.close(); },
            "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none(), nb::lock_self(),
            nb::sig("def __exit__(self, exc_type: type[BaseException] | None, exc_value: "
                    "BaseException | None, traceback: types.TracebackType | None) -> None"));

    nb::class_<PreparedStatement>(m, "PreparedStatement", "A reusable compiled SQL statement.")
        .def("execute", &PreparedStatement::execute, "parameters"_a = nb::none(),
             "streaming"_a = false,
             nb::sig("def execute(self, parameters: list | tuple | dict[str, typing.Any] | None = "
                     "None, streaming: bool = False) -> Result"),
             "Bind parameters and return the result. Streaming results are single-pass.",
             nb::lock_self())
        .def("executemany", &PreparedStatement::executemany, "parameters"_a,
             nb::sig("def executemany(self, parameters: collections.abc.Iterable[list | tuple | "
                     "dict[str, typing.Any]]) -> None"),
             "Run once per parameter set and discard the results.", nb::lock_self())
        .def_prop_ro("num_parameters", &PreparedStatement::num_parameters,
                     "Number of bind parameters.", nb::lock_self())
        .def("parameter_name", &PreparedStatement::parameter_name, "index"_a,
             nb::sig("def parameter_name(self, index: int) -> str | None"),
             "Name of the parameter at 1-based ``index``, or None for positional / "
             "out-of-range.",
             nb::lock_self())
        .def_prop_ro("columns", &PreparedStatement::column_names,
                     "Result column names, known ahead of execution (empty for "
                     "non-result statements).",
                     nb::lock_self())
        .def_prop_ro("types", &PreparedStatement::column_types,
                     "Result column type names, known ahead of execution.", nb::lock_self())
        .def_prop_ro("statement_type", &PreparedStatement::statement_type,
                     nb::sig("def statement_type(self) -> str"),
                     "Statement kind: ``'SELECT'``, ``'INSERT'``, ``'UPDATE'``, ...",
                     nb::lock_self())
        .def("close", &PreparedStatement::close, "Destroy the prepared statement.", nb::lock_self())
        .def(
            "__enter__", [](PreparedStatement& self) -> PreparedStatement& { return self; },
            nb::rv_policy::reference, nb::sig("def __enter__(self) -> typing.Self"))
        .def(
            "__exit__",
            [](PreparedStatement& self, nb::object, nb::object, nb::object) { self.close(); },
            "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none(),
            nb::sig("def __exit__(self, exc_type: type[BaseException] | None, exc_value: "
                    "BaseException | None, traceback: types.TracebackType | None) -> None"),
            nb::lock_self());

    nb::enum_<duckdb_pending_state>(m, "PendingState",
                                    "State of an async query's executor after a tick.")
        .value("READY", DUCKDB_PENDING_RESULT_READY, "The result is ready to materialize.")
        .value("NOT_READY", DUCKDB_PENDING_RESULT_NOT_READY, "More tasks remain; tick again.")
        .value("ERROR", DUCKDB_PENDING_ERROR, "Execution failed; see ``.error()``.")
        .value("NO_TASKS", DUCKDB_PENDING_NO_TASKS_AVAILABLE,
               "Workers own the outstanding tasks; yield, then tick again.");

    nb::class_<PendingResult>(m, "PendingResult", "Steppable in-flight query handle.")
        .def("execute_task", &PendingResult::execute_task,
             nb::sig("def execute_task(self) -> PendingState"),
             "Advance the executor by one task (GIL released) and return the new state.")
        .def(
            "execute_tasks", &PendingResult::execute_tasks, "budget"_a = 0.005,
            nb::sig("def execute_tasks(self, budget: float = 0.005) -> PendingState"),
            "Run tasks for up to ``budget`` seconds (GIL released), stopping once ready or failed.")
        .def("error", &PendingResult::error, nb::sig("def error(self) -> str"),
             "DuckDB's message for the most recent error, may be empty.")
        .def("materialize", &PendingResult::materialize, nb::sig("def materialize(self) -> Result"),
             "Materialize the finished result; call once ``execute_task`` reports ``READY``.")
        .def("drain", &PendingResult::drain, nb::sig("def drain(self) -> None"),
             "Interrupt the query and drain the executor (cancellation teardown).");

    nb::class_<Connection> conn_cls(m, "Connection", nb::type_slots(connection_slots),
                                    "A connection to a DuckDB database.");
    conn_cls
        .def("execute", &Connection::execute, "query"_a, "parameters"_a = nb::none(),
             "streaming"_a = false, nb::rv_policy::reference,
             nb::sig("def execute(self, query: str, parameters: list | tuple | "
                     "dict[str, typing.Any] | None = None, streaming: bool = False) -> Connection"),
             "Execute one statement and retain its result for fetch methods. Streaming "
             "results are single-pass. Returns ``self``.",
             nb::lock_self())
        .def("executemany", &Connection::executemany, "query"_a, "parameters"_a,
             nb::sig("def executemany(self, query: str, parameters: "
                     "collections.abc.Iterable[list | tuple | dict[str, typing.Any]]) -> None"),
             "Run once per parameter set and discard the results.", nb::lock_self())
        .def(
            "sql",
            [](Connection& self, const std::string& query, bool streaming) {
                return self.sql(query, nb::none(), streaming);
            },
            "query"_a, "streaming"_a = false,
            nb::sig("def sql(self, query: str, streaming: bool = False) -> Result"),
            "Run a query and return its result directly.", nb::lock_self())
        .def(
            "query",
            [](Connection& self, const std::string& query, bool streaming) {
                return self.sql(query, nb::none(), streaming);
            },
            "query"_a, "streaming"_a = false,
            nb::sig("def query(self, query: str, streaming: bool = False) -> Result"),
            "Alias for :meth:`sql`.", nb::lock_self())
        .def("_query", &Connection::sql, "query"_a, "parameters"_a = nb::none(),
             "streaming"_a = false,
             nb::sig("def _query(self, query: str, parameters: list | tuple | "
                     "dict[str, typing.Any] | None = None, streaming: bool = False) -> Result"),
             "Like :meth:`sql`, with parameters.", nb::lock_self())
        .def("_arrow_collected_stream", &Connection::arrow_collected_stream, "query"_a,
             "parameters"_a = nb::none(), "rows_per_batch"_a = 1000000,
             nb::sig("def _arrow_collected_stream(self, query: str, parameters: list | tuple | "
                     "dict[str, typing.Any] | None = None, rows_per_batch: int = 1000000) "
                     "-> typing.Any"),
             "Return the query as an Arrow C stream capsule.", nb::lock_self())
        .def(
            "aexecute",
            [](nb::object self, const std::string& query, nb::object parameters, bool streaming) {
                return aio().attr("aexecute")(self, query, parameters, streaming);
            },
            "query"_a, "parameters"_a = nb::none(), "streaming"_a = false,
            nb::sig("def aexecute(self, query: str, parameters: list | tuple | "
                    "dict[str, typing.Any] | None = None, streaming: bool = False) "
                    "-> collections.abc.Coroutine[typing.Any, typing.Any, Result]"),
            "Execute asynchronously and return a Result without setting current_result.")
        .def(
            "asql",
            [](nb::object self, const std::string& query, bool streaming) {
                return aio().attr("asql")(self, query, streaming);
            },
            "query"_a, "streaming"_a = false,
            nb::sig("def asql(self, query: str, streaming: bool = False) "
                    "-> collections.abc.Coroutine[typing.Any, typing.Any, Result]"),
            "Async variant of :meth:`sql`. See :meth:`aexecute`.")
        .def("make_pending", &Connection::make_pending, "query"_a, "parameters"_a = nb::none(),
             "streaming"_a = false,
             nb::sig("def make_pending(self, query: str, parameters: list | tuple | "
                     "dict[str, typing.Any] | None = None, streaming: bool = False) "
                     "-> PendingResult"),
             "Build a steppable pending result.", nb::lock_self())
        .def("prepare", &Connection::prepare, "query"_a,
             nb::sig("def prepare(self, query: str) -> PreparedStatement"),
             "Compile a reusable prepared statement.", nb::lock_self())
        .def("fetchone", &Connection::fetchone, nb::sig("def fetchone(self) -> tuple | None"),
             "Fetch one row from the latest execute result.", nb::lock_self())
        .def("fetchmany", &Connection::fetchmany, "size"_a = 1,
             nb::sig("def fetchmany(self, size: int = 1) -> list[tuple]"),
             "Fetch up to ``size`` rows from the latest result.", nb::lock_self())
        .def("fetchall", &Connection::fetchall, nb::sig("def fetchall(self) -> list[tuple]"),
             "Fetch all remaining rows from the latest result.", nb::lock_self())
        .def("fetchitem", &Connection::fetchitem, nb::sig("def fetchitem(self) -> typing.Any"),
             "Return the lone scalar from the latest 1x1 result.", nb::lock_self())
        .def_prop_ro(
            "rows_changed", [](Connection& self) { return self.current_result()->rows_changed(); },
            "Rows affected by the latest INSERT, UPDATE, or DELETE.", nb::lock_self())
        .def_prop_ro("description", &Connection::description,
                     nb::sig("def description(self) -> list[tuple] | None"),
                     "PEP 249 result description for the most recent query, or None.",
                     nb::lock_self())
        .def_prop_ro("columns", &Connection::columns,
                     nb::sig("def columns(self) -> list[str] | None"),
                     "Column names of the most recent result, or None.", nb::lock_self())
        .def_prop_ro("current_result", &Connection::current_result,
                     nb::sig("def current_result(self) -> Result"),
                     "The latest execute result; raises if no query has run.", nb::lock_self());
    def_conversions(conn_cls, [](nb::object self) {
        return nb::cast(nb::cast<Connection&>(self).current_result());
    });
    conn_cls.def(
        "arrow",
        [](nb::object self, const std::string& query, nb::object parameters) {
            return conversions().attr("arrow_query")(self, query, parameters);
        },
        "query"_a, "parameters"_a = nb::none(),
        nb::sig("def arrow(self, query: str, parameters: list | tuple | "
                "dict[str, typing.Any] | None = None) -> pyarrow.Table"),
        "Execute a query and return its complete Arrow table.");
    conn_cls.def(
        "arrow_reader",
        [](nb::object self, const std::string& query, nb::object parameters,
           uint64_t rows_per_batch) {
            return conversions().attr("arrow_reader")(self, query, parameters, rows_per_batch);
        },
        "query"_a, "parameters"_a = nb::none(), "rows_per_batch"_a = 1000000,
        nb::sig("def arrow_reader(self, query: str, parameters: list | tuple | "
                "dict[str, typing.Any] | None = None, rows_per_batch: int = 1000000) -> "
                "pyarrow.RecordBatchReader"),
        "Execute a query and stream batches of up to ``rows_per_batch`` rows.");
    conn_cls.def(
        "to_numpy",
        [](nb::object self, nb::object columns) {
            return nb::cast<Connection&>(self).current_result()->to_numpy(columns);
        },
        "columns"_a = nb::none(),
        nb::sig("def to_numpy(self, columns: collections.abc.Iterable[str] | None = None) "
                "-> dict[str, numpy.ndarray]"),
        "Drain the latest result into a {name: numpy.ndarray} dict. "
        "Raises on VARCHAR/LIST/STRUCT/MAP; use arrow() for those.");
    conn_cls
        .def(
            "create_function",
            [](Connection& self, const std::string& name, nb::callable fn, nb::object parameters,
               nb::object return_type, nb::object varargs, nb::object init, bool is_volatile,
               bool special_handling) {
                create_scalar_function(self, name, std::move(fn), parameters, return_type, varargs,
                                       init, is_volatile, special_handling);
            },
            "name"_a, "fn"_a, "parameters"_a = nb::none(), "return_type"_a = nb::none(),
            "varargs"_a = nb::none(), "init"_a = nb::none(), nb::kw_only(), "volatile"_a = false,
            "special_handling"_a = false,
            nb::sig("def create_function(self, name: str, fn: collections.abc.Callable, "
                    "parameters: list[str] | dict[str, str] | None = None, "
                    "return_type: str | None = None, "
                    "varargs: str | None = None, "
                    "init: collections.abc.Callable[[], typing.Any] | None = None, "
                    "*, volatile: bool = False, special_handling: bool = False) -> None"),
            "Register a chunk-wise NumPy scalar UDF. Types may be explicit or inferred. "
            "Rows with a NULL argument are not passed to fn and yield NULL; with "
            "special_handling, inputs and output use ``(values, validity)`` tuples instead.",
            nb::lock_self())
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
            "Register an Arrow scalar UDF. The callable receives Arrow arrays, a named "
            "array dict, or a RecordBatch, and returns an Arrow array.",
            nb::lock_self())
        .def(
            "create_aggregate_function",
            [](Connection& self, const std::string& name, nb::object cls, nb::object parameters,
               const std::string& return_type) {
                create_aggregate_function(self, name, cls, parameters, return_type);
            },
            "name"_a, "cls"_a, "parameters"_a, "return_type"_a,
            nb::sig("def create_aggregate_function(self, name: str, cls: type, "
                    "parameters: list[str] | dict[str, str], return_type: str) -> None"),
            "Register an aggregate class defining update, combine and finalize. Rows with "
            "a NULL argument are skipped. combine(other) merges another partial state "
            "into self without modifying other.",
            nb::lock_self())
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
            "Register a table function yielding rows matching ``columns``.", nb::lock_self())
        .def(
            "appender",
            [](Connection& self, const std::string& table, std::optional<std::string> schema,
               std::optional<std::string> catalog) {
                return new Appender(self, table, std::move(schema), std::move(catalog));
            },
            "table"_a, "schema"_a = nb::none(), "catalog"_a = nb::none(),
            nb::sig("def appender(self, table: str, schema: str | None = None, "
                    "catalog: str | None = None) -> Appender"),
            "Open a bulk appender for ``table``.", nb::lock_self())
        .def("interrupt", &Connection::interrupt,
             "Request cancellation of a running query; thread-safe.")
        .def("progress", &Connection::progress,
             nb::sig("def progress(self) -> tuple[float, int, int]"),
             "Return ``(percentage, rows_processed, total_rows)``; percentage is -1 "
             "until estimated.")
        .def("get_profiling_info", &Connection::get_profiling_info,
             nb::sig("def get_profiling_info(self) -> dict[str, typing.Any] | None"),
             "Return the latest profiling tree, or None when profiling is disabled.")
        .def("set_profile_sink", &Connection::set_profile_sink, "sink"_a.none(), nb::kw_only(),
             "sample"_a = 1, "mode"_a = "standard",
             nb::sig("def set_profile_sink(self, sink: collections.abc.Callable[[str, "
                     "dict[str, typing.Any]], None] | None, *, sample: int = 1, "
                     "mode: str = 'standard') -> None"),
             "Call ``sink(query, info)`` after every Nth materialized query. Pass None "
             "to detach. Streaming queries are skipped.",
             nb::lock_self())
        .def("register_arrow", &Connection::register_arrow, "name"_a, "obj"_a,
             nb::sig("def register_arrow(self, name: str, obj: typing.Any) -> None"),
             "Register an Arrow C stream source as table ``name`` (matched "
             "case-insensitively). Tables and data frames are re-read by every query; "
             "readers and query results are one-shot and can be scanned once.",
             nb::lock_self())
        .def("load_extension", &Connection::load_extension, "name"_a,
             nb::sig("def load_extension(self, name: str) -> None"),
             "Load an extension by name or path. Unsigned extensions require "
             "allow_unsigned_extensions=True when connecting.",
             nb::lock_self())
        .def("close", &Connection::close, "Close the connection.", nb::lock_self())
        .def(
            "__enter__", [](Connection& self) -> Connection& { return self; },
            nb::rv_policy::reference, nb::sig("def __enter__(self) -> typing.Self"))
        .def(
            "__exit__", [](Connection& self, nb::object, nb::object, nb::object) { self.close(); },
            "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none(),
            nb::sig("def __exit__(self, exc_type: type[BaseException] | None, exc_value: "
                    "BaseException | None, traceback: types.TracebackType | None) -> None"),
            nb::lock_self());

    m.def(
        "connect",
        [](const std::string& database, nb::object config) {
            return new Connection(database, std::move(config));
        },
        "database"_a = ":memory:", "config"_a = nb::none(),
        nb::sig("def connect(database: str = ':memory:', config: dict[str, str] | None = None) "
                "-> Connection"),
        "Open a database with an optional string-valued configuration.");

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
        "Return ``(name, description)`` for every DuckDB setting accepted by "
        ":func:`connect`.");
}
