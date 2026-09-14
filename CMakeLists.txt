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

# Instrument DuckDB and the extension with AddressSanitizer.
option(DUCKY_ASAN "Build with AddressSanitizer" OFF)
if(DEFINED ENV{DUCKY_ASAN} AND NOT DUCKY_ASAN)
  set(DUCKY_ASAN ON)
endif()
if(DUCKY_ASAN)
  add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address)
endif()

# Python + nanobind
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

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

# Disable DuckDB's standalone tools by default.
set(BUILD_SHELL OFF CACHE BOOL "")
set(BUILD_UNITTESTS OFF CACHE BOOL "")
set(BUILD_BENCHMARKS OFF CACHE BOOL "")

# picky records a version because the shallow submodule cannot describe itself.
set(_describe "")
  if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/duckdb-describe.txt")
      file(STRINGS "${CMAKE_SOURCE_DIR}/cmake/duckdb-describe.txt" _describe LIMIT_COUNT 1)
  endif()
  set(OVERRIDE_GIT_DESCRIBE "${_describe}" CACHE STRING "" FORCE)

# Build and statically link the extensions required at runtime.
set(BUILD_EXTENSIONS "httpfs;json;icu" CACHE STRING "" FORCE)
set(DUCKDB_STATIC_EXTENSIONS core_functions parquet httpfs json icu)

add_subdirectory(${DUCKDB_DIR} duckdb EXCLUDE_FROM_ALL)

# The extension module
# Opt into free-threaded Python and retain Release optimization for conversion loops.
nanobind_add_module(
  _core
  NB_STATIC
  FREE_THREADED
  NOMINSIZE
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
target_link_libraries(_core PRIVATE duckdb_static tsl::robin_map)

# Preserve DuckDB's generated extension loader through dead stripping.
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

# Prevent dllimport annotations when statically linking on Windows.
target_compile_definitions(_core PRIVATE DUCKDB_STATIC_BUILD)

# Target flags let stub generation preload the ASAN runtime.
if(DUCKY_ASAN)
  target_compile_options(_core PRIVATE -fsanitize=address -fno-omit-frame-pointer)
  target_link_options(_core PRIVATE -fsanitize=address)
endif()

install(TARGETS _core LIBRARY DESTINATION ducky)

# Generate in the build tree; explicit targets update or verify the source stub.
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

# Format generated stubs when Ruff is available.
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
