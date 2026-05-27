#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "connection.hpp"
#include "ducky.hpp"
#include "duckdb.h"
#include "result.hpp"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_core, m) {
    m.doc() = "ducky: tiny nanobind bindings for the DuckDB C API";
    m.attr("__duckdb_version__") = duckdb_library_version();

    // C++ DuckyError -> Python ducky.Error.
    nb::exception<DuckyError>(m, "Error");

    nb::class_<Result>(m, "Result",
                       "A query result. Iterate it, or use the fetch* methods, "
                       "to pull rows as tuples.")
        .def_prop_ro("columns", &Result::column_names, "Column names.")
        .def_prop_ro("types", &Result::column_types, "Column type names.")
        .def_prop_ro("description", &Result::description,
                     "PEP 249 result description.")
        .def("fetchone", &Result::fetchone, "Return the next row, or None.")
        .def("fetchmany", &Result::fetchmany, "size"_a = 1,
             "Return up to `size` rows.")
        .def("fetchall", &Result::fetchall, "Return all remaining rows.")
        .def(
            "__arrow_c_stream__",
            [](nb::object self, nb::object) { return nb::cast<Result &>(self).arrow_c_stream(self); },
            "requested_schema"_a = nb::none(),
            "Export the result via the Arrow C stream (PyCapsule) interface.")
        .def(
            "arrow",
            [](nb::object self) {
                return nb::module_::import_("ducky._conversions").attr("arrow")(self);
            },
            "Return the result as a pyarrow.Table.")
        .def(
            "df",
            [](nb::object self) {
                return nb::module_::import_("ducky._conversions").attr("df")(self);
            },
            "Return the result as a pandas.DataFrame.")
        .def(
            "pl",
            [](nb::object self, bool lazy) {
                return nb::module_::import_("ducky._conversions").attr("pl")(self, lazy);
            },
            "lazy"_a = false, "Return the result as a polars DataFrame (or LazyFrame).")
        .def(
            "fetchnumpy",
            [](nb::object self) {
                return nb::module_::import_("ducky._conversions").attr("fetchnumpy")(self);
            },
            "Return the result as a dict of column name -> numpy array.")
        .def("__iter__", [](nb::object self) { return self; })
        .def("__next__", [](Result &self) {
            nb::object row = self.fetchone();
            if (row.is_none()) throw nb::stop_iteration();
            return row;
        });

    nb::class_<Connection>(m, "Connection", "A connection to a DuckDB database.")
        .def("execute", &Connection::execute, "query"_a,
             "parameters"_a = nb::none(), nb::rv_policy::reference,
             "Execute a query, optionally with positional parameters, and "
             "return self for chaining.")
        // The returned Result shares the DuckDBHandle, so it keeps the database
        // open on its own — no keep_alive needed.
        .def("sql", &Connection::sql, "query"_a, "Run a query and return its Result.")
        .def("query", &Connection::sql, "query"_a, "Alias for sql().")
        .def("fetchone", &Connection::fetchone)
        .def("fetchmany", &Connection::fetchmany, "size"_a = 1)
        .def("fetchall", &Connection::fetchall)
        .def_prop_ro("description", &Connection::description)
        .def_prop_ro("columns", &Connection::columns)
        .def(
            "arrow",
            [](Connection &self) {
                return nb::module_::import_("ducky._conversions")
                    .attr("arrow")(nb::cast(self.current_result()));
            },
            "Return the last result as a pyarrow.Table.")
        .def(
            "df",
            [](Connection &self) {
                return nb::module_::import_("ducky._conversions")
                    .attr("df")(nb::cast(self.current_result()));
            },
            "Return the last result as a pandas.DataFrame.")
        .def(
            "pl",
            [](Connection &self, bool lazy) {
                return nb::module_::import_("ducky._conversions")
                    .attr("pl")(nb::cast(self.current_result()), lazy);
            },
            "lazy"_a = false, "Return the last result as a polars DataFrame (or LazyFrame).")
        .def(
            "fetchnumpy",
            [](Connection &self) {
                return nb::module_::import_("ducky._conversions")
                    .attr("fetchnumpy")(nb::cast(self.current_result()));
            },
            "Return the last result as a dict of column name -> numpy array.")
        .def("close", &Connection::close, "Close the connection.")
        .def("__enter__", [](Connection &self) -> Connection & { return self; },
             nb::rv_policy::reference)
        .def(
            "__exit__",
            [](Connection &self, nb::object, nb::object, nb::object) {
                self.close();
            },
            "exc_type"_a.none(), "exc_value"_a.none(), "traceback"_a.none(),
            nb::sig("def __exit__(self, exc_type: type[BaseException] | None, exc_value: "
                    "BaseException | None, traceback: types.TracebackType | None) -> None"));

    m.def(
        "connect",
        [](const std::string &database) { return new Connection(database); },
        "database"_a = ":memory:",
        "Open `database` (default in-memory) and return a Connection.");
}
