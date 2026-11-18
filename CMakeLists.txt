cmake_minimum_required(VERSION 3.29)

project(ducky LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Generate a compilation database for each build tree. Developer builds can
# copy theirs to the repository root without letting isolated package builds
# replace it with paths from a temporary environment.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
option(DUCKY_SYNC_COMPILE_COMMANDS
  "Copy compile_commands.json to the source root" OFF)
if(PROJECT_IS_TOP_LEVEL AND DUCKY_SYNC_COMPILE_COMMANDS)
  add_custom_target(
    sync_compile_commands ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_BINARY_DIR}/compile_commands.json
      ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
    BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
    COMMENT "Syncing compile_commands.json to source root"
    VERBATIM)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# DuckDB is instrumented too, since misused buffers and callbacks that ducky
# hands it fail inside DuckDB.
option(DUCKY_ASAN "Build with AddressSanitizer" OFF)
option(DUCKY_UBSAN "Build with UndefinedBehaviorSanitizer" OFF)
option(DUCKY_TSAN "Build with ThreadSanitizer" OFF)
if(DUCKY_TSAN AND DUCKY_ASAN)
  message(FATAL_ERROR "DUCKY_TSAN and DUCKY_ASAN cannot be enabled at the same time")
endif()
set(_ducky_sanitize_compile "")
set(_ducky_sanitize_link "")
if(DUCKY_ASAN)
  list(APPEND _ducky_sanitize_compile -fsanitize=address)
  list(APPEND _ducky_sanitize_link -fsanitize=address)
endif()
# Fail on the first finding rather than letting a test pass after undefined behavior.
if(DUCKY_UBSAN)
  list(APPEND _ducky_sanitize_compile -fsanitize=undefined -fno-sanitize-recover=undefined)
  list(APPEND _ducky_sanitize_link -fsanitize=undefined)
endif()
if(DUCKY_TSAN)
  list(APPEND _ducky_sanitize_compile -fsanitize=thread)
  list(APPEND _ducky_sanitize_link -fsanitize=thread)
endif()
if(_ducky_sanitize_compile)
  list(APPEND _ducky_sanitize_compile -fno-omit-frame-pointer)
  add_compile_options(${_ducky_sanitize_compile})
  add_link_options(${_ducky_sanitize_link})
endif()
# DuckDB adds ASan and UBSan to its own Debug builds, which would mix
# sanitizers whose runtimes the process does not load.
set(ENABLE_SANITIZER OFF CACHE BOOL "" FORCE)
set(ENABLE_UBSAN OFF CACHE BOOL "" FORCE)
set(ENABLE_THREAD_SANITIZER OFF CACHE BOOL "" FORCE)

# Keep symbol names for profilers and stack traces, at the cost of a larger
# module. Sanitizer reports need them too.
option(DUCKY_SYMBOLS "Keep symbol names in the extension module" OFF)
if((DEFINED ENV{DUCKY_SYMBOLS} OR _ducky_sanitize_compile) AND NOT DUCKY_SYMBOLS)
  set(DUCKY_SYMBOLS ON)
endif()

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# DuckDB is built from the sparse submodule through its C API.
set(DUCKDB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/duckdb")

if(NOT EXISTS "${DUCKDB_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "ext/duckdb is empty. Run: picky init ext/duckdb")
endif()

# Only the library is needed.
set(BUILD_SHELL OFF CACHE BOOL "")
set(BUILD_UNITTESTS OFF CACHE BOOL "")

# picky records a version because the shallow submodule cannot describe itself.
set(_describe "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/duckdb-describe.txt")
  file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/duckdb-describe.txt" _describe LIMIT_COUNT 1)
endif()
set(OVERRIDE_GIT_DESCRIBE "${_describe}" CACHE STRING "" FORCE)

# Build and statically link the extensions required at runtime.
set(BUILD_EXTENSIONS "httpfs;json;icu" CACHE STRING "" FORCE)
set(DUCKDB_STATIC_EXTENSIONS core_functions parquet httpfs json icu)

add_subdirectory(${DUCKDB_DIR} duckdb EXCLUDE_FROM_ALL)

# Opt into free-threaded Python and retain Release optimization for conversion loops.
set(_ducky_module_options NB_STATIC FREE_THREADED NOMINSIZE)
if(DUCKY_SYMBOLS)
  list(APPEND _ducky_module_options NOSTRIP)
endif()
nanobind_add_module(
  _core
  ${_ducky_module_options}
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

# Link the extensions, plus the HTTP client and loadable-extension support that
# INSTALL and LOAD need, ahead of DuckDB itself. DuckDB generates a loader that
# registers them in this order, which is also their load order.
set(DUCKDB_MODULE_BASE_DIR "${DUCKDB_DIR}")
duckdb_link_extensions(
  _core PRIVATE httplib loadable_extensions ${DUCKDB_STATIC_EXTENSIONS})
target_link_libraries(_core PRIVATE duckdb_static tsl::robin_map)

# Prevent dllimport annotations when statically linking on Windows, and declare
# the unstable C API v2, whose shape the DuckDB pin fixes.
target_compile_definitions(_core PRIVATE DUCKDB_STATIC_BUILD DUCKDB_V2_API_ALLOW_UNSTABLE=1)

# The statically linked DuckDB would otherwise export its C++ symbols, and the
# loader would bind another DuckDB in the process (duckdb-python's, say) to
# ours: macOS coalesces weak definitions across images, Linux unifies
# STB_GNU_UNIQUE ones. Loadable extensions link their own copy of DuckDB,
# unless built with EXTENSION_STATIC_BUILD=OFF.
if(APPLE)
  target_link_options(_core PRIVATE "LINKER:-exported_symbol,_PyInit__core")
  # Hidden symbols otherwise lose their names, even without stripping.
  if(DUCKY_SYMBOLS)
    target_link_options(_core PRIVATE "LINKER:-keep_private_externs")
  endif()
elseif(UNIX)
  set(_ducky_exports "${CMAKE_CURRENT_BINARY_DIR}/_core.exports")
  file(WRITE "${_ducky_exports}" "{ global: PyInit__core; local: *; };\n")
  target_link_options(_core PRIVATE "LINKER:--version-script=${_ducky_exports}")
  set_property(TARGET _core APPEND PROPERTY LINK_DEPENDS "${_ducky_exports}")
endif()

# Target flags let stub generation preload the sanitizer runtime.
if(_ducky_sanitize_compile)
  target_compile_options(_core PRIVATE ${_ducky_sanitize_compile})
  target_link_options(_core PRIVATE ${_ducky_sanitize_link})
endif()

install(TARGETS _core LIBRARY DESTINATION ducky)

# Generated in the build tree, so package builds never touch the checked-in stub.
# Explicit targets update or verify it.
set(_DUCKY_CORE_STUB "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi")
nanobind_add_stub(
    ducky_core_stub
    MODULE _core
    OUTPUT ${_DUCKY_CORE_STUB}
    PYTHON_PATH $<TARGET_FILE_DIR:_core>
    PATTERN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/stubgen.pat
    DEPENDS _core
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/ducky/_core.pyi DESTINATION ducky)

# Formatted so the checked-in stub passes the repository's ruff hook.
execute_process(
    COMMAND ${Python_EXECUTABLE} -m ruff --version
    RESULT_VARIABLE _ruff_available
    OUTPUT_QUIET ERROR_QUIET
)
if(_ruff_available EQUAL 0)
    add_custom_command(
        TARGET ducky_core_stub POST_BUILD
        COMMAND ${Python_EXECUTABLE} -m ruff check --fix --exit-zero --quiet
            ${_DUCKY_CORE_STUB}
        COMMAND ${Python_EXECUTABLE} -m ruff format --quiet
            ${_DUCKY_CORE_STUB}
        COMMENT "ruff fix + format _core.pyi"
        VERBATIM
    )
endif()

add_custom_target(
    update_ducky_core_stub
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${_DUCKY_CORE_STUB}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ducky/_core.pyi
    DEPENDS ducky_core_stub
    COMMENT "Updating checked-in src/ducky/_core.pyi"
    VERBATIM)

add_custom_target(
    check_ducky_core_stub
    COMMAND ${CMAKE_COMMAND} -E compare_files
        ${_DUCKY_CORE_STUB}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ducky/_core.pyi
    DEPENDS ducky_core_stub
    COMMENT "Checking checked-in src/ducky/_core.pyi"
    VERBATIM)
