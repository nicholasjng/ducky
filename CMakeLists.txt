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

# AddressSanitizer build. Driven by -DDUCKY_ASAN=ON (or DUCKY_ASAN=1 in the env
# from scikit-build-core). The flag goes onto the _core target's COMPILE_OPTIONS
# / LINK_OPTIONS (not CMAKE_CXX_FLAGS) so nanobind's stubgen sanitizer-detector
# (nanobind PR #1000) finds it and preloads libclang_rt.asan via DYLD_INSERT_LIBRARIES
# when importing _core during stub generation — putting the flag on the global
# CMAKE_CXX_FLAGS works for the compiler but not for the detector, which walks
# per-target properties only. add_compile_options also covers duckdb_static so
# its objects stay ASAN-instrumented.
option(DUCKY_ASAN "Build with AddressSanitizer" OFF)
if(DEFINED ENV{DUCKY_ASAN} AND NOT DUCKY_ASAN)
  set(DUCKY_ASAN ON)
endif()
if(DUCKY_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

# ──────────────────────────────────────────────────────────────────────────────
# Python + nanobind
# ──────────────────────────────────────────────────────────────────────────────
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

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
# FREE_THREADED opts the module into no-GIL execution on free-threaded CPython
# (3.13t+): nanobind emits Py_MOD_GIL_NOT_USED so importing it doesn't re-enable
# the GIL. It's a no-op on regular (GIL-enabled) interpreters. See the
# free-threading item in ROADMAP.md for the scaling rationale + audit status.
nanobind_add_module(
  _core
  NB_STATIC
  FREE_THREADED
  src/cpp/ducky.cpp
  src/cpp/appender.cpp
  src/cpp/connection.cpp
  src/cpp/chunk.cpp
  src/cpp/function.cpp
  src/cpp/aggregate.cpp
  src/cpp/table.cpp
  src/cpp/arrow_scan.cpp
  src/cpp/result.cpp)

target_include_directories(_core PRIVATE ${DUCKDB_DIR}/src/include)
# tsl::robin_map is vendored by nanobind under ext/robin_map/include and exposed
# as an IMPORTED target by nanobind-config.cmake; link it so the headers are
# visible to our sources (nanobind itself only links it PRIVATE to its own
# static lib).
target_link_libraries(_core PRIVATE duckdb_static tsl::robin_map)

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

# Repeat -fsanitize=address on the _core target itself: add_compile_options sets
# the *directory*'s COMPILE_OPTIONS, but nanobind's stubgen sanitizer detector
# (PR #1000, see nanobind_sanitizer_preload_env in nanobind-config.cmake) only
# scans per-target COMPILE_OPTIONS / LINK_OPTIONS. Without this, the detector
# concludes "no sanitizer" and runs stubgen without preloading libclang_rt.asan,
# which then aborts at `import _core` with "Interceptors are not working".
if(DUCKY_ASAN)
  target_compile_options(_core PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(_core PRIVATE -fsanitize=address)
endif()

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
    PATTERN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/stub_patterns.txt
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
        COMMAND ${Python_EXECUTABLE} -m ruff check --fix --exit-zero --quiet
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ducky/_core.pyi
        COMMAND ${Python_EXECUTABLE} -m ruff format --quiet
            ${CMAKE_CURRENT_SOURCE_DIR}/src/ducky/_core.pyi
        COMMENT "ruff fix + format _core.pyi"
        VERBATIM
    )
endif()
