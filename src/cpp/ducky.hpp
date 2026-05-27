#pragma once

#include <stdexcept>

// Exception thrown across the binding layer. Registered with nanobind in
// ducky.cpp so it surfaces in Python as `ducky.Error`.
struct DuckyError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
