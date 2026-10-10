# SPDX-License-Identifier: GPL-3.0-or-later
#
# Builds the ctypes bridge as a shared library on top of the EXISTING custom-ops
# target.  It does not edit research/custom-ops; add_subdirectory consumes its
# CMakeLists as-is and links the static `custom_ops` / `co_schedule` targets.
#
# Requires the pinned upstream checkouts (research/upstream-components) as described
# in research/upstream-components/README.md.
#
#   cmake -S research/multidevice-scheduling/backends/bridge -B build/co-backend-bridge
#   cmake --build build/co-backend-bridge -j4
#
# The result is libco_backend_bridge.so; point CO_BACKEND_BRIDGE at it, or let the
# adapter search the default build directory.

cmake_minimum_required(VERSION 3.16)
project(co_backend_bridge LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# The static custom-ops targets are linked into a shared object here, so their
# objects must be position independent.  Setting this project-wide is what keeps us
# from editing research/custom-ops/CMakeLists.txt.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CUSTOM_OPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../custom-ops")
if(NOT EXISTS "${CUSTOM_OPS_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "custom-ops sources not found at ${CUSTOM_OPS_DIR}")
endif()

add_subdirectory("${CUSTOM_OPS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/custom-ops")

add_library(co_backend_bridge SHARED co_backend_bridge.c)
target_include_directories(co_backend_bridge PRIVATE "${CUSTOM_OPS_DIR}/include")
target_link_libraries(co_backend_bridge PRIVATE custom_ops co_schedule)
target_compile_options(co_backend_bridge PRIVATE -Wall -Wextra)
set_target_properties(co_backend_bridge PROPERTIES OUTPUT_NAME co_backend_bridge)
