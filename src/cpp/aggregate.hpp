#pragma once

#include <nanobind/nanobind.h>

#include <string>
#include <vector>

#include "function.hpp"

namespace nb = nanobind;

class Connection;

struct AggregateUDFContext {
    nb::callable cls;
    std::vector<const TypeSpec*> param_types;
    const TypeSpec* return_type;
    bool has_combine;
};

void create_aggregate_function(Connection& con, const std::string& name, nb::object cls,
                               nb::object parameters, const std::string& return_type);
