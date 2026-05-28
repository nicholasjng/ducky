cmake_minimum_required(VERSION 3.29)

project(ducky LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile_commands.json for clangd / IDE tooling (see CLAUDE.md). The
# file lands in ${CMAKE_BINARY_DIR}; we also symlink it to the source root so
# clangd (and IDEs that probe the repo root) pick it up without configuration.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_custom_target(
  symlink_compile_commands ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_BINARY_DIR}/compile_commands.json
    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
  COMMENT "Symlinking compile_commands.json to source root"
  VERBATIM)

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

if(NOT EXISTS "${DUCKDB_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "ext/duckdb is empty. Run: git submodule update --init ext/duckdb")
endif()

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
# parquet — in ext/duckdb/extension/extension_config.cmake, and the generated
# loader references them from the DuckDB constructor. To pull in additional
# in-tree extensions, list them via BUILD_EXTENSIONS: DuckDB will call
# duckdb_extension_load() on each, which builds and registers the loader.
# We must then also link each one (see the link step below). Leave
# DISABLE_BUILTIN_EXTENSIONS at its default (OFF): turning it on makes the
# loader reference core_functions without building it, which fails to link.
#
# Note: out-of-tree extensions (httpfs, azure, …) live in separate DuckDB-org
# repos. DuckDB's build system fetches them at configure time when listed in
# BUILD_EXTENSIONS — it reads .github/config/extensions/<name>.cmake which
# contains the GIT_URL/GIT_TAG. The sparse-checkout pattern must include the
# matching .cmake file (and any .github/patches/extensions/<name>/ entry).
set(BUILD_EXTENSIONS "httpfs;json" CACHE STRING "" FORCE)
set(DUCKDB_STATIC_EXTENSIONS core_functions parquet httpfs json)

# Out-of-tree extensions (httpfs, azure, …) take DuckDB's APPLY_PATCHES path,
# which hardcodes ${CMAKE_SOURCE_DIR}/scripts/apply_extension_patches.py and
# ${CMAKE_SOURCE_DIR}/.github/patches/extensions/<ext>/ instead of using the
# DuckDB submodule's own DUCKDB_MODULE_BASE_DIR. Stage symlinks into our
# source root so the build finds them. The patches keep the upstream httpfs
# repo buildable against the DuckDB commit we vendored — skipping them isn't
# safe across DuckDB version bumps.
file(MAKE_DIRECTORY
     "${CMAKE_CURRENT_SOURCE_DIR}/scripts"
     "${CMAKE_CURRENT_SOURCE_DIR}/.github/patches/extensions")
foreach(_link IN ITEMS
        "scripts/apply_extension_patches.py"
        ".github/patches/extensions/httpfs")
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_link}")
    file(CREATE_LINK "${DUCKDB_DIR}/${_link}"
                     "${CMAKE_CURRENT_SOURCE_DIR}/${_link}" SYMBOLIC)
  endif()
endforeach()

add_subdirectory(${DUCKDB_DIR} duckdb EXCLUDE_FROM_ALL)

# ──────────────────────────────────────────────────────────────────────────────
# The extension module
# ──────────────────────────────────────────────────────────────────────────────
nanobind_add_module(
  _core
  NB_STATIC
  src/cpp/ducky.cpp
  src/cpp/connection.cpp
  src/cpp/chunk.cpp
  src/cpp/function.cpp
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
