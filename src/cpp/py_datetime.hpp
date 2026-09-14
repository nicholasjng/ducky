#pragma once

#include <nanobind/nanobind.h>
// datetime.h needs Python.h (via nanobind) included first.
#include <datetime.h>

#include <cstdint>

#include "duckdb.h"

// Per-translation-unit Python/DuckDB datetime conversions.

namespace pydt {

[[maybe_unused]] static duckdb_date date_to_duckdb(PyObject* raw) {
    duckdb_date_struct s{};
    s.year = PyDateTime_GET_YEAR(raw);
    s.month = (int8_t)PyDateTime_GET_MONTH(raw);
    s.day = (int8_t)PyDateTime_GET_DAY(raw);
    return duckdb_to_date(s);
}

[[maybe_unused]] static duckdb_time time_to_duckdb(PyObject* raw) {
    duckdb_time_struct s{};
    s.hour = (int8_t)PyDateTime_TIME_GET_HOUR(raw);
    s.min = (int8_t)PyDateTime_TIME_GET_MINUTE(raw);
    s.sec = (int8_t)PyDateTime_TIME_GET_SECOND(raw);
    s.micros = PyDateTime_TIME_GET_MICROSECOND(raw);
    return duckdb_to_time(s);
}

// Unix microseconds; aware values are normalized to UTC.
[[maybe_unused]] static int64_t datetime_to_micros(nanobind::handle val) {
    PyObject* raw = val.ptr();
    duckdb_timestamp_struct s{};
    s.date.year = PyDateTime_GET_YEAR(raw);
    s.date.month = (int8_t)PyDateTime_GET_MONTH(raw);
    s.date.day = (int8_t)PyDateTime_GET_DAY(raw);
    s.time.hour = (int8_t)PyDateTime_DATE_GET_HOUR(raw);
    s.time.min = (int8_t)PyDateTime_DATE_GET_MINUTE(raw);
    s.time.sec = (int8_t)PyDateTime_DATE_GET_SECOND(raw);
    s.time.micros = PyDateTime_DATE_GET_MICROSECOND(raw);
    int64_t micros = duckdb_to_timestamp(s).micros;
    nanobind::object offset = val.attr("utcoffset")();
    if (!offset.is_none()) {
        PyObject* d = offset.ptr();
        int64_t off = ((int64_t)PyDateTime_DELTA_GET_DAYS(d) * 86400 +
                       (int64_t)PyDateTime_DELTA_GET_SECONDS(d)) *
                          1000000 +
                      (int64_t)PyDateTime_DELTA_GET_MICROSECONDS(d);
        micros -= off;
    }
    return micros;
}

}  // namespace pydt
