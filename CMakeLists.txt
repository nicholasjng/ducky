cmake_minimum_required(VERSION 3.29)

project(ducky LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile_commands.json for clangd / IDE tooling (see CLAUDE.md).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ──────────────────────────────────────────────────────────────────────────────
# Python + nanobind
# ──────────────────────────────────────────────────────────────────────────────
find_package(Python 3.13 REQUIRED COMPONENTS Interpreter Development.Module)

# Locate the installed nanobind package and import its CMake config.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# ──────────────────────────────────────────────────────────────────────────────
# DuckDB, built from the checked-out submodule (ext/duckdb -> duckdb/duckdb). We
# bind the stable C API only.
# ──────────────────────────────────────────────────────────────────────────────
set(DUCKDB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/duckdb")

# Trim DuckDB down to what a Python extension needs. These are plain cache
# defaults (no FORCE), so they can still be overridden at configure time.
set(BUILD_SHELL OFF CACHE BOOL "")
set(BUILD_UNITTESTS OFF CACHE BOOL "")
set(BUILD_BENCHMARKS OFF CACHE BOOL "")

# The ext/duckdb submodule is a shallow clone with no tags, so DuckDB's own
# `git describe` version detection falls back to "v0.0.1". Pin the version string
# to the describe of the pinned commit so duckdb_library_version() is accurate.
# Update this if you move the submodule to a different commit.
set(OVERRIDE_GIT_DESCRIBE "v1.5.2-4685-g01eda16d6e" CACHE STRING "")

# DuckDB unconditionally loads its "essential" extensions — core_functions and
# parquet — in ext/duckdb/extension/extension_config.cmake, and the
# generated loader references them from the DuckDB constructor. We must link
# exactly that set (see the link step below). Leave DISABLE_BUILTIN_EXTENSIONS
# at its default (OFF): turning it on makes the loader reference core_functions
# without building it, which fails to link.
set(DUCKDB_STATIC_EXTENSIONS core_functions parquet)

add_subdirectory(${DUCKDB_DIR} duckdb EXCLUDE_FROM_ALL)

# ──────────────────────────────────────────────────────────────────────────────
# The extension module
# ──────────────────────────────────────────────────────────────────────────────
nanobind_add_module(
  _core
  NB_STATIC
  src/cpp/ducky.cpp
  src/cpp/connection.cpp
  src/cpp/result.cpp)

target_include_directories(_core PRIVATE ${DUCKDB_DIR}/src/include)
target_link_libraries(_core PRIVATE duckdb_static)

# duckdb_static's DuckDB constructor calls ExtensionHelper::LoadAllExtensions,
# which is defined in the generated loader rather than in duckdb_static itself.
# Pull it in WHOLE_ARCHIVE so the linker keeps it despite -dead_strip. Because
# DuckDB is added EXCLUDE_FROM_ALL, each *_extension target only builds if we
# link it, so we link exactly the set the loader references.
if(TARGET duckdb_generated_extension_loader)
  target_link_libraries(
    _core PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,duckdb_generated_extension_loader>")
endif()
foreach(ext IN LISTS DUCKDB_STATIC_EXTENSIONS)
  if(TARGET ${ext}_extension)
    target_link_libraries(_core PRIVATE ${ext}_extension)
  endif()
endforeach()

# Without this, duckdb.h marks the C API symbols as dllimport on Windows,
# producing unresolved __imp_* references when linking the static library.
# No-op on other platforms.
target_compile_definitions(_core PRIVATE DUCKDB_STATIC_BUILD)

install(TARGETS _core LIBRARY DESTINATION ducky)

# Generate a .pyi stub for the compiled module alongside the Python package
# source so IDEs and type checkers (Zed/ty/Pyright) pick it up without
# inspecting the installed wheel — the src/-layout otherwise wins over
# site-packages when resolving `ducky._core`. The stub is regenerated on every
# build and committed to version control; scikit-build-core packs it into
# the wheel automatically via `wheel.packages`.
nanobind_add_stub(
    _core_stub
    MODULE _core
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/ducky/_core.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_core>
    DEPENDS _core
)

# Run `ruff format` on the freshly-generated stub so it matches the rest of
# the codebase's style. Skipped silently if ruff isn't available (e.g. when
# building from an sdist without dev extras).
execute_process(
    COMMAND ${Python_EXECUTABLE} -m ruff --version
    RESULT_VARIABLE _ruff_available
    OUTPUT_QUIET ERROR_QUIET
)
if(_ruff_available EQUAL 0)
    add_custom_command(
        TARGET _core_stub POST_BUILD
        COMMAND ${Python_EXECUTABLE} -m ruff check --fix --quiet
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ducky/_core.pyi
        COMMAND ${Python_EXECUTABLE} -m ruff format --quiet
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ducky/_core.pyi
        COMMENT "ruff fix + format _core.pyi"
        VERBATIM
    )
endif()
