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

// Cached module handles — imported once, intentionally leaked. See
// cached_singleton() in ducky.hpp.
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

// Register the DataFrame/array/Arrow accessors shared by Result and Connection.
// `src` maps the bound object to the Arrow/chunk source — identity for Result,
// `.current_result()` for Connection. Defining them here once keeps the two
// surfaces in lockstep and lets stubgen emit them.
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
        "Return the result as a dict of column name -> numpy array. Numeric / "
        "temporal columns only; use arrow() for strings / nested types.");
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
        "iter_batches_mlx",
        [src](nb::object self, nb::object columns) {
            return conversions().attr("iter_batches_mlx")(src(self), columns);
        },
        "columns"_a = nb::none(),
        nb::sig("def iter_batches_mlx(self, columns: collections.abc.Iterable[str] | None = "
                "None) -> collections.abc.Iterator[dict[str, mlx.core.array]]"),
        "Yield one dict per chunk: {name: mlx.core.array}. Zero-copy on CPU via DLPack.");
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
    cls.def(
        "to_mlx",
        [src](nb::object self, nb::object columns) {
            return conversions().attr("to_mlx")(src(self), columns);
        },
        "columns"_a = nb::none(),
        nb::sig("def to_mlx(self, columns: collections.abc.Iterable[str] | None = None) "
                "-> dict[str, mlx.core.array]"),
        "Eagerly concatenate all chunks into {name: mlx.core.array}.");
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "ducky: tiny nanobind bindings for the DuckDB C API";
    m.attr("__duckdb_version__") = duckdb_library_version();

    // C++ DuckyError -> Python ducky.Error.
    // NOLINTNEXTLINE(bugprone-unused-raii)
    nb::exception<DuckyError>(m, "Error");

    nb::class_<Chunk>(m, "Chunk",
                      "A column-major batch of up to STANDARD_VECTOR_SIZE rows.\n\n"
                      "Numeric and temporal columns are exposed as zero-copy ndarrays "
                      "over DuckDB's internal buffers.")
        .def("__len__", &Chunk::size, "Number of rows.")
        .def_prop_ro("columns", &Chunk::column_names, "Column names.")
        .def_prop_ro("types", &Chunk::column_types, "Column type names.")
        .def(
            "column",
            [](nb::object self, nb::object key) {
                return nb::cast<Chunk&>(self).column(key, self);
            },
            "key"_a, nb::sig("def column(self, key: int | str) -> numpy.ndarray"),
            "Return the column at `key` as a 1-D ndarray view.\n\n"
            "Parameters\n"
            "----------\n"
            "key : int or str\n"
            "    Column index or name.")
        .def(
            "validity",
            [](nb::object self, nb::object key) {
                return nb::cast<Chunk&>(self).validity(key, self);
            },
            "key"_a, nb::sig("def validity(self, key: int | str) -> numpy.ndarray | None"),
            "Return a uint8 validity mask (1 = valid, 0 = NULL), or None if no nulls.")
        .def("decimal_scale", &Chunk::decimal_scale, "key"_a,
             nb::sig("def decimal_scale(self, key: int | str) -> int"),
             "Return the scale (digits after the decimal point) of a DECIMAL column.")
        .def(
            "dlpack",
            [](nb::object self, nb::object key) {
                return nb::cast<Chunk&>(self).dlpack(key, self);
            },
            "key"_a, nb::sig("def dlpack(self, key: int | str) -> typing.Any"),
            "Return the column as a DLPack object (``__dlpack__`` / "
            "``__dlpack_device__``), without going through numpy.\n"
            "Flat numeric / temporal types only.");

    nb::class_<Result> result_cls(m, "Result",
                                  "A query result. Iterate or call ``fetch*`` to pull rows.");
    result_cls.def_prop_ro("columns", &Result::column_names, "Column names.")
        .def_prop_ro("types", &Result::column_types, "Column type names.")
        .def_prop_ro("description", &Result::description,
                     nb::sig("def description(self) -> list[tuple] | None"),
                     "PEP 249 result description, or None if the result is closed.")
        // nb::lock_self() serializes cursor mutation on free-threaded builds;
        // no-op under the GIL.
        .def("fetchone", &Result::fetchone, nb::sig("def fetchone(self) -> tuple | None"),
             "Return the next row as a tuple, or None at end of result.", nb::lock_self())
        .def("fetchmany", &Result::fetchmany, "size"_a = 1,
             nb::sig("def fetchmany(self, size: int = 1) -> list[tuple]"),
             "Return up to ``size`` rows.", nb::lock_self())
        .def("fetchall", &Result::fetchall, nb::sig("def fetchall(self) -> list[tuple]"),
             "Return all remaining rows.", nb::lock_self())
        .def("fetchitem", &Result::fetchitem, nb::sig("def fetchitem(self) -> typing.Any"),
             "Return the lone scalar of a 1-row × 1-column result.\n\n"
             "Useful for ``COUNT(*)``-style queries."
             "Raises :class:`Error` if the result is not exactly one row and one column.",
             nb::lock_self())
        .def("fetch_chunk", &Result::fetch_chunk, nb::sig("def fetch_chunk(self) -> Chunk | None"),
             "Pull the next :class:`Chunk`, or None at end of stream.", nb::lock_self())
        .def("to_numpy", &Result::to_numpy, "columns"_a = nb::none(),
             nb::sig("def to_numpy(self, columns: collections.abc.Iterable[str] "
                     "| None = None) -> dict[str, numpy.ndarray]"),
             "Drain the result into a {name: numpy.ndarray} dict. "
             "Raises on VARCHAR/LIST/STRUCT/MAP — use .arrow() for those.",
             nb::lock_self())
        .def(
            "__arrow_c_stream__",
            [](nb::object self, nb::object) {
                return nb::cast<Result&>(self).arrow_c_stream(self);
            },
            "requested_schema"_a = nb::none(),
            nb::sig("def __arrow_c_stream__(self, requested_schema: typing.Any = None) "
                    "-> typing.Any"),
            "Export the result via the Arrow PyCapsule stream interface.");
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
            nb::sig("def __next__(self) -> tuple"), nb::lock_self());

    nb::class_<Appender>(m, "Appender",
                         "Bulk-insert handle for a target table.\n\n"
                         "Supports row-at-a-time inserts, a columnar fast path for "
                         "numeric / temporal ndarrays, and an Arrow ingest path that "
                         "covers VARCHAR / LIST / STRUCT.")
        .def_prop_ro("columns", &Appender::column_names, "Target column names.")
        .def_prop_ro("types", &Appender::column_types, "Target column type names.")
        // nb::lock_self() serializes buffered-chunk mutation on free-threaded
        // builds; no-op under the GIL.
        .def("append_row", &Appender::append_row,
             nb::sig("def append_row(self, *values: object) -> None"),
             "Append one row of positional values, in target column order.", nb::lock_self())
        .def("append_columns", &Appender::append_columns, "columns"_a, "masks"_a = nb::none(),
             nb::sig("def append_columns(self, columns: dict[str, numpy.ndarray], "
                     "masks: dict[str, numpy.ndarray] | None = None) -> None"),
             "Append a batch of named columns; missing columns are filled with NULL.\n\n"
             "Parameters\n"
             "----------\n"
             "columns : dict[str, numpy.ndarray]\n"
             "    Column-name → 1-D ndarray of equal length.\n"
             "masks : dict[str, numpy.ndarray], optional\n"
             "    Per-column validity (1-D bool / uint8; True = valid).",
             nb::lock_self())
        .def("append_arrow", &Appender::append_arrow, "source"_a,
             nb::sig("def append_arrow(self, source: typing.Any) -> None"),
             "Append from any object exposing the Arrow PyCapsule interface.\n\n"
             "Accepts ``__arrow_c_stream__`` (pyarrow Table / RecordBatchReader / "
             "polars / pandas-3 / a ducky :class:`Result`) or ``__arrow_c_array__`` "
             "(a pyarrow RecordBatch).\n"
             "Columns must line up positionally with the target table and match its types.",
             nb::lock_self())
        .def("flush", &Appender::flush, "Flush pending rows to the table.", nb::lock_self())
        .def("close", &Appender::close, "Flush and tear down the appender.", nb::lock_self())
        .def(
            "__enter__", [](Appender& self) -> Appender& { return self; }, nb::rv_policy::reference)
        .def(
            "__exit__", [](Appender& self, nb::object, nb::object, nb::object) { self.close(); },
            "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none(), nb::lock_self(),
            nb::sig("def __exit__(self, exc_type: type[BaseException] | None, exc_value: "
                    "BaseException | None, traceback: types.TracebackType | None) -> None"));

    nb::class_<PreparedStatement>(
        m, "PreparedStatement",
        "A SQL statement compiled once via :meth:`Connection.prepare`.\n\n"
        "Executable repeatedly with different parameters without re-parsing.")
        // nb::lock_self() everywhere — methods mutate stmt_, which close()
        // frees. No-op under the GIL.
        .def("execute", &PreparedStatement::execute, "parameters"_a = nb::none(),
             "streaming"_a = false,
             nb::sig("def execute(self, parameters: list | tuple | dict[str, typing.Any] | None = "
                     "None, streaming: bool = False) -> Result"),
             "Bind ``parameters`` and run the statement, returning its :class:`Result`.\n\n"
             "Parameters\n"
             "----------\n"
             "parameters : list | tuple | dict[str, Any], optional\n"
             "    Positional sequence or named dict; ``None`` runs without binding.\n"
             "streaming : bool, default False\n"
             "    Return a streaming result whose chunks are pulled lazily (peak "
             "memory stays bounded to one chunk).\n"
             "    Single-pass; consume via ``fetch*`` or ``iter_batches_*``, not both.",
             nb::lock_self())
        .def("executemany", &PreparedStatement::executemany, "parameters"_a,
             nb::sig("def executemany(self, parameters: collections.abc.Iterable[list | tuple | "
                     "dict[str, typing.Any]]) -> None"),
             "Run the statement once per parameter set, discarding each result.\n\n"
             "The fast path for batched ``INSERT`` / ``UPDATE`` / ``DELETE``.",
             nb::lock_self())
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
                     "Statement kind: ``'SELECT'``, ``'INSERT'``, ``'UPDATE'``, ….",
                     nb::lock_self())
        .def("close", &PreparedStatement::close, "Destroy the prepared statement.", nb::lock_self())
        .def(
            "__enter__", [](PreparedStatement& self) -> PreparedStatement& { return self; },
            nb::rv_policy::reference)
        .def(
            "__exit__",
            [](PreparedStatement& self, nb::object, nb::object, nb::object) { self.close(); },
            "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none(),
            nb::sig("def __exit__(self, exc_type: type[BaseException] | None, exc_value: "
                    "BaseException | None, traceback: types.TracebackType | None) -> None"),
            nb::lock_self());

    // Async substrate driven by ducky._aio; end users go through aexecute / asql.
    nb::enum_<duckdb_pending_state>(m, "PendingState",
                                    "State of an async query's executor after a tick.")
        .value("READY", DUCKDB_PENDING_RESULT_READY, "The result is ready to materialize.")
        .value("NOT_READY", DUCKDB_PENDING_RESULT_NOT_READY, "More tasks remain; tick again.")
        .value("ERROR", DUCKDB_PENDING_ERROR, "Execution failed; see ``.error()``.")
        .value("NO_TASKS", DUCKDB_PENDING_NO_TASKS_AVAILABLE,
               "Workers own the outstanding tasks; yield, then tick again.");

    nb::class_<PendingResult>(
        m, "PendingResult",
        "Steppable handle over an in-flight async query.\n\n"
        "Internal: built by :meth:`Connection.make_pending` and driven by the "
        "``ducky._aio`` event-loop helpers.\n"
        "Prefer :meth:`Connection.aexecute` / :meth:`Connection.asql`.")
        .def("execute_task", &PendingResult::execute_task,
             nb::sig("def execute_task(self) -> PendingState"),
             "Advance the executor by one task (GIL released) and return the new state.")
        .def("error", &PendingResult::error, nb::sig("def error(self) -> str"),
             "DuckDB's message for the most recent error, may be empty.")
        .def("materialize", &PendingResult::materialize, nb::sig("def materialize(self) -> Result"),
             "Materialize the finished result; call once ``execute_task`` reports ``READY``.")
        .def("drain", &PendingResult::drain, nb::sig("def drain(self) -> None"),
             "Interrupt the query and drain the executor (cancellation teardown).");

    nb::class_<Connection> conn_cls(m, "Connection",
                                    "A connection to a DuckDB database.\n\n"
                                    "Each :func:`connect` returns one connection that owns its own "
                                    "database handle; two ``connect(\":memory:\")`` calls are "
                                    "isolated.");
    conn_cls
        .def("execute", &Connection::execute, "query"_a, "parameters"_a = nb::none(),
             "streaming"_a = false, nb::rv_policy::reference,
             nb::sig("def execute(self, query: str, parameters: list | tuple | "
                     "dict[str, typing.Any] | None = None, streaming: bool = False) -> Connection"),
             "Execute a SQL query and stash the result for the ``fetch*`` methods.\n\n"
             "Parameters\n"
             "----------\n"
             "query : str\n"
             "    SQL text.\n"
             "    Single statement only.\n"
             "parameters : list | tuple | dict[str, Any], optional\n"
             "    Positional sequence or named dict; ``None`` runs without binding.\n"
             "streaming : bool, default False\n"
             "    Return a streaming result whose chunks are pulled lazily; useful "
             "for ``iter_batches_*`` on large queries.\n"
             "    Single-pass; consume via ``fetch*`` or ``iter_batches_*``, not both.\n\n"
             "Returns\n"
             "-------\n"
             "Connection\n"
             "    ``self``, so calls can be chained (e.g. "
             "``con.execute(sql).fetchall()``).\n\n"
             "Examples\n"
             "--------\n"
             ">>> rows = con.execute(\"SELECT * FROM t WHERE id = ?\", [42]).fetchall()",
             nb::lock_self())
        // The returned Result shares the DuckDBHandle, so no keep_alive needed.
        .def("sql", &Connection::sql, "query"_a, "streaming"_a = false,
             nb::sig("def sql(self, query: str, streaming: bool = False) -> Result"),
             "Run a query and return its :class:`Result` directly.\n\n"
             "Unlike :meth:`execute`, ``sql`` does not stash the result on the connection.\n"
             "See :meth:`execute` for ``streaming``.",
             nb::lock_self())
        .def("query", &Connection::sql, "query"_a, "streaming"_a = false,
             nb::sig("def query(self, query: str, streaming: bool = False) -> Result"),
             "Alias for :meth:`sql`.", nb::lock_self())
        .def(
            "aexecute",
            [](nb::object self, const std::string& query, nb::object parameters, bool streaming) {
                return aio().attr("aexecute")(self, query, parameters, streaming);
            },
            "query"_a, "parameters"_a = nb::none(), "streaming"_a = false,
            nb::sig("def aexecute(self, query: str, parameters: list | tuple | "
                    "dict[str, typing.Any] | None = None, streaming: bool = False) "
                    "-> collections.abc.Coroutine[typing.Any, typing.Any, Result]"),
            "Async variant of :meth:`execute`.\n\n"
            "Ticks the executor one task at a time off-thread so the event loop stays "
            "responsive; cancelling the await interrupts the query.\n"
            "Resolves to the :class:`Result` directly — does not set ``current_result``.")
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
             "Build a steppable :class:`PendingResult` for the ``ducky._aio`` drivers.\n"
             "Low-level; prefer :meth:`aexecute` / :meth:`asql`.",
             nb::lock_self())
        .def("prepare", &Connection::prepare, "query"_a,
             nb::sig("def prepare(self, query: str) -> PreparedStatement"),
             "Compile ``query`` once into a :class:`PreparedStatement` for repeated "
             "execution, avoiding per-call re-parsing and planning.",
             nb::lock_self())
        // lock_self() guards last_result_ against a concurrent swap.
        .def("fetchone", &Connection::fetchone, nb::sig("def fetchone(self) -> tuple | None"),
             "Fetch one row from the most recent :meth:`execute` result. Delegates to "
             ":meth:`Result.fetchone`.",
             nb::lock_self())
        .def("fetchmany", &Connection::fetchmany, "size"_a = 1,
             nb::sig("def fetchmany(self, size: int = 1) -> list[tuple]"),
             "Fetch up to ``size`` rows. Delegates to :meth:`Result.fetchmany`.", nb::lock_self())
        .def("fetchall", &Connection::fetchall, nb::sig("def fetchall(self) -> list[tuple]"),
             "Fetch all remaining rows. Delegates to :meth:`Result.fetchall`.", nb::lock_self())
        .def("fetchitem", &Connection::fetchitem, nb::sig("def fetchitem(self) -> typing.Any"),
             "Return the lone scalar of a 1-row × 1-column result.\n"
             "Delegates to :meth:`Result.fetchitem`.",
             nb::lock_self())
        .def_prop_ro("description", &Connection::description,
                     nb::sig("def description(self) -> list[tuple] | None"),
                     "PEP 249 result description for the most recent query, or None.",
                     nb::lock_self())
        .def_prop_ro("columns", &Connection::columns,
                     nb::sig("def columns(self) -> list[str] | None"),
                     "Column names of the most recent result, or None.", nb::lock_self())
        .def_prop_ro("current_result", &Connection::current_result,
                     nb::sig("def current_result(self) -> Result"),
                     "The most recent :class:`Result` from :meth:`execute`.\n\n"
                     "Raises :class:`Error` if no query has run yet.\n"
                     "The conversion accessors (``arrow`` / ``df`` / ``pl`` / …) "
                     "delegate to this.",
                     nb::lock_self());
    // Connection's conversion accessors operate on its most recent result.
    def_conversions(conn_cls, [](nb::object self) {
        return nb::cast(nb::cast<Connection&>(self).current_result());
    });
    // to_numpy is bound directly (not via def_conversions) so it doesn't
    // detour through the Python conversions module — the C++ method runs the
    // chunk loop without leaving the binding.
    conn_cls.def(
        "to_numpy",
        [](nb::object self, nb::object columns) {
            return nb::cast<Connection&>(self).current_result()->to_numpy(columns);
        },
        "columns"_a = nb::none(),
        nb::sig("def to_numpy(self, columns: collections.abc.Iterable[str] | None = None) "
                "-> dict[str, numpy.ndarray]"),
        "Drain the latest result into a {name: numpy.ndarray} dict. "
        "Raises on VARCHAR/LIST/STRUCT/MAP — use .arrow() for those.");
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
            "Register a Python callable as a DuckDB scalar UDF.\n\n"
            "Inputs arrive as zero-copy 1-D ndarrays; ``fn`` must return one "
            "ndarray of matching dtype.\n\n"
            "Parameters\n"
            "----------\n"
            "name : str\n"
            "    Name to register the function under.\n"
            "fn : callable\n"
            "    Called as ``fn(*args)`` (positional ``parameters``) or "
            "``fn(kwargs)`` (dict ``parameters``).\n"
            "parameters : list[str] | dict[str, str], optional\n"
            "    DuckDB type strings — list for positional, dict for keyword.\n"
            "    If omitted, types are inferred from ``fn``'s annotations "
            "(``bool``/``int``/``float`` → ``BOOLEAN``/``BIGINT``/``DOUBLE``).\n"
            "return_type : str, optional\n"
            "    Inferred from annotations if omitted.\n"
            "varargs : str, optional\n"
            "    DuckDB type for a variable-arity function.\n"
            "    Mutually exclusive with ``parameters``.\n"
            "init : callable, optional\n"
            "    Zero-arg factory for per-worker-thread state.\n"
            "    Its return value becomes the first positional arg of ``fn`` on "
            "every chunk.\n"
            "volatile : bool, default False\n"
            "    Mark non-deterministic (e.g. RNG, clock) so the optimizer "
            "won't fold or cache it.\n"
            "special_handling : bool, default False\n"
            "    NULL-aware: arguments arrive as ``(values, mask)`` tuples "
            "(``mask`` is a 1-D uint8 ndarray, 1 = valid) and the UDF emits "
            "NULLs via the same return shape.",
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
            "Register a scalar UDF backed by the Arrow C-API path.\n\n"
            "Supports VARCHAR, LIST, STRUCT, DECIMAL, MAP — anything DuckDB can "
            "round-trip through Arrow.\n"
            "``fn`` is called with one ``pyarrow.Array`` per column (positional, or "
            "``{name: Array}`` dict if ``parameters`` is a dict); set "
            "``record_batch=True`` to receive one ``pyarrow.RecordBatch`` instead.\n"
            "Must return a ``pyarrow.Array``.",
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
            "Register a Python class as a DuckDB aggregate UDF.\n\n"
            "``cls`` must define ``__init__``, ``update(self, *arrays)``, and "
            "``finalize(self) -> scalar``.\n"
            "An optional ``combine(self, other)`` enables parallel aggregate execution.",
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
            "Register a Python generator as a DuckDB table function.\n\n"
            "``factory`` is called with the SQL arguments and must return a "
            "generator that yields one tuple per row.\n"
            "``columns`` is an ordered dict ``{column_name: type_string}`` "
            "declaring the output schema.",
            nb::lock_self())
        .def(
            "appender",
            [](Connection& self, const std::string& table, std::optional<std::string> schema,
               std::optional<std::string> catalog) {
                return new Appender(self, table, std::move(schema), std::move(catalog));
            },
            "table"_a, "schema"_a = nb::none(), "catalog"_a = nb::none(),
            nb::sig("def appender(self, table: str, schema: str | None = None, "
                    "catalog: str | None = None) -> Appender"),
            "Open an :class:`Appender` for bulk inserts into ``table``.\n\n"
            "The appender shares the connection's database handle, so the "
            "database stays open until the appender is closed.",
            nb::lock_self())
        // interrupt() / progress() use thread-safe DuckDB C calls and are
        // meant to run from another thread while execute() holds the lock —
        // no lock_self() on them.
        .def("interrupt", &Connection::interrupt,
             "Best-effort cancel of any query running on this connection.\n\n"
             "Safe to call from another thread while :meth:`execute` is in flight.")
        .def("progress", &Connection::progress,
             nb::sig("def progress(self) -> tuple[float, int, int]"),
             "Snapshot of the current query's progress.\n\n"
             "Returns\n"
             "-------\n"
             "tuple[float, int, int]\n"
             "    ``(percentage, rows_processed, total_rows_to_process)``.\n"
             "    ``percentage`` is ``-1.0`` until DuckDB has an estimate.")
        .def("get_profiling_info", &Connection::get_profiling_info,
             nb::sig("def get_profiling_info(self) -> dict[str, typing.Any] | None"),
             "Programmatic EXPLAIN ANALYZE for the most recent query.\n\n"
             "Returns a nested ``{\"metrics\": {str: str}, \"children\": [...]}`` "
             "dict, or ``None`` if profiling isn't enabled.\n"
             "For ergonomics use :func:`ducky.profile` (context manager) or "
             ":func:`ducky.format_profiling_info` (pretty-printer).\n"
             "Metric values are strings per the DuckDB C API; coerce numerics on "
             "the caller side.")
        .def("set_profile_sink", &Connection::set_profile_sink, "sink"_a.none(), nb::kw_only(),
             "sample"_a = 1, "mode"_a = "standard",
             nb::sig("def set_profile_sink(self, sink: collections.abc.Callable[[str, "
                     "dict[str, typing.Any]], None] | None, *, sample: int = 1, "
                     "mode: str = 'standard') -> None"),
             "Install an always-on profile sink for every :meth:`execute` / "
             ":meth:`sql` on this connection.\n\n"
             "Parameters\n"
             "----------\n"
             "sink : callable or None\n"
             "    Called as ``sink(query, info)`` after every materialized query "
             "with the same nested dict :meth:`get_profiling_info` returns.\n"
             "    Pass ``None`` to detach.\n"
             "sample : int, default 1\n"
             "    Fire the sink every Nth query (useful in tight training loops).\n"
             "mode : str, default 'standard'\n"
             "    DuckDB ``profiling_mode``. ``'detailed'`` adds per-operator "
             "counters like ``cpu_time``.\n\n"
             "Notes\n"
             "-----\n"
             "Streaming results (``streaming=True``) are skipped — their chunks "
             "haven't been pulled when ``execute`` returns.\n"
             "Sink exceptions are reported via ``PyErr_WriteUnraisable`` and do "
             "not propagate.",
             nb::lock_self())
        .def("register_arrow", &Connection::register_arrow, "name"_a, "obj"_a,
             nb::sig("def register_arrow(self, name: str, obj: typing.Any) -> None"),
             "Register an Arrow-PyCapsule source as a table named ``name``.\n\n"
             "Accepts anything exposing ``__arrow_c_stream__`` (pyarrow Table, "
             "polars / pandas-3 DataFrame, …).\n"
             "Lazy and zero-copy: the source is kept and re-streamed on each query "
             "via a replacement scan, so it must support being streamed more than "
             "once.",
             nb::lock_self())
        .def("close", &Connection::close, "Close the connection.", nb::lock_self())
        .def(
            "__enter__", [](Connection& self) -> Connection& { return self; },
            nb::rv_policy::reference)
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
        "Open ``database`` (default in-memory) and return a :class:`Connection`.\n\n"
        "Low-level entrypoint.\n"
        "The Python wrapper :func:`ducky.connect` accepts DuckDB settings as typed "
        "keyword arguments and is what most users want.\n\n"
        "Parameters\n"
        "----------\n"
        "database : str, default ':memory:'\n"
        "    File path, ``':memory:'``, or any DuckDB connection string.\n"
        "config : dict[str, str], optional\n"
        "    DuckDB settings applied at open time.\n"
        "    See :func:`config_options`.");

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
