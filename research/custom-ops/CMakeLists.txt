cmake_minimum_required(VERSION 3.16)
project(rk3588_custom_ops LANGUAGES C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(UPSTREAM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../upstream-components" CACHE PATH "Pinned upstream checkouts")
set(RKT "${UPSTREAM_ROOT}/rocket-userspace")
if(NOT EXISTS "${RKT}/src/npu_regcmd.c")
  message(FATAL_ERROR "Restore upstream-components using its provenance.json first")
endif()
set(ROCKETNPU_INCLUDE_DIR "${RKT}/include" CACHE PATH "" FORCE)
add_subdirectory("${UPSTREAM_ROOT}/rknpu-submit" provider)
# Only the portable generator and submit seam are needed; no frontend or ARM NEON dependency.
add_library(custom_ops STATIC src/custom_ops.c src/linear_hardsigmoid.c src/linear_plan.c src/co_graph.c src/co_row_ops.c src/co_rope.c src/co_attention.c src/co_decoder.c src/co_gdn.c src/co_gdn_cpu.c
  "${RKT}/src/npu_regcmd.c" "${RKT}/src/rocket_hw_profile.c"
  "${RKT}/src/rocket_sysfs.c" "${RKT}/src/rocket_log.c"
  "${RKT}/src/rocket_chain.c" "${RKT}/src/rocket_op.c")
target_include_directories(custom_ops PUBLIC include "${RKT}/include" PRIVATE src "${RKT}/src")
target_link_libraries(custom_ops PUBLIC rknpu-submit m)
target_compile_options(custom_ops PRIVATE -Wall -Wextra)
add_executable(custom_ops_probe tools/c/probe.c)
target_link_libraries(custom_ops_probe PRIVATE custom_ops)
target_compile_options(custom_ops_probe PRIVATE -Wall -Wextra -Werror)
add_executable(linear_hardsigmoid tools/c/operator_cli.c)
target_link_libraries(linear_hardsigmoid PRIVATE custom_ops)
target_compile_options(linear_hardsigmoid PRIVATE -Wall -Wextra -Werror)
enable_testing()
add_test(NAME custom_ops_host COMMAND custom_ops_probe host)
set_tests_properties(custom_ops_host PROPERTIES ENVIRONMENT "ROCKET_CHIP=rk3588")
add_executable(ktiling_host_test tests/c/ktiling_host_test.c)
target_link_libraries(ktiling_host_test PRIVATE custom_ops)
target_compile_options(ktiling_host_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME ktiling_host COMMAND ktiling_host_test)
set_tests_properties(ktiling_host PROPERTIES ENVIRONMENT "ROCKET_CHIP=rk3588")
add_test(NAME custom_operator_cli_help COMMAND linear_hardsigmoid --help)
add_executable(two_task_probe tools/c/two_task_probe.c)
target_include_directories(two_task_probe PRIVATE src "${RKT}/src")
target_link_libraries(two_task_probe PRIVATE custom_ops dl)
target_compile_options(two_task_probe PRIVATE -Wall -Wextra -Werror)
add_executable(ktiling_probe tools/c/ktiling_probe.c)
target_include_directories(ktiling_probe PRIVATE src "${RKT}/src")
target_link_libraries(ktiling_probe PRIVATE custom_ops dl)
target_compile_options(ktiling_probe PRIVATE -Wall -Wextra -Werror)
add_library(two_task_trace SHARED tools/c/two_task_trace.c)
target_include_directories(two_task_trace PRIVATE "${UPSTREAM_ROOT}/rknpu-submit/src" include "${RKT}/include")
target_compile_definitions(two_task_trace PRIVATE CO_MAX_WORDS=2048)
target_link_libraries(two_task_trace PRIVATE dl)
target_compile_options(two_task_trace PRIVATE -Wall -Wextra)
add_test(NAME two_task_host COMMAND two_task_probe host)
add_test(NAME ktiling_probe_host COMMAND ktiling_probe host)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME two_task_evidence_negative COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/two_task_verify_test.py")
add_library(co_schedule STATIC src/co_schedule.c)
target_include_directories(co_schedule PUBLIC include PRIVATE "${RKT}/src" "${UPSTREAM_ROOT}/rknpu-submit/src")
target_link_libraries(co_schedule PUBLIC custom_ops)
target_compile_options(co_schedule PRIVATE -Wall -Wextra -Werror)
add_executable(schedule_probe tools/c/schedule_probe.c)
target_link_libraries(schedule_probe PRIVATE co_schedule)
target_compile_options(schedule_probe PRIVATE -Wall -Wextra -Werror)
add_executable(overlap_probe tools/c/overlap_probe.c)
target_include_directories(overlap_probe PRIVATE src "${RKT}/src")
target_link_libraries(overlap_probe PRIVATE co_schedule pthread)
target_link_options(overlap_probe PRIVATE -rdynamic)
target_compile_options(overlap_probe PRIVATE -Wall -Wextra -Werror)
add_test(NAME overlap_host COMMAND overlap_probe host)
set_tests_properties(overlap_host PROPERTIES ENVIRONMENT "ROCKET_CHIP=rk3588")
add_test(NAME overlap_evidence_negative COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/overlap_verify_test.py")
add_test(NAME schedule_host COMMAND schedule_probe host)
add_test(NAME schedule_evidence_negative COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/schedule_analyze_test.py")
add_test(NAME device_cost_schema COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/device_cost_schema_test.py")
add_executable(cost_probe tools/c/cost_probe.c)
target_link_libraries(cost_probe PRIVATE co_schedule)
target_compile_options(cost_probe PRIVATE -Wall -Wextra -Werror)
add_executable(interference_load tools/c/interference_load.c)
target_link_libraries(interference_load PRIVATE pthread)
target_compile_options(interference_load PRIVATE -Wall -Wextra -Werror)
add_test(NAME cost_probe_help COMMAND cost_probe --help)
add_executable(multitile_host_test tests/c/multitile_host_test.c)
target_link_libraries(multitile_host_test PRIVATE custom_ops)
target_compile_options(multitile_host_test PRIVATE -Wall -Wextra -Werror)
target_link_options(multitile_host_test PRIVATE
  -Wl,--wrap=rocket_bo_alloc32 -Wl,--wrap=rocket_bo_free
  -Wl,--wrap=rocket_bo_prep -Wl,--wrap=rocket_bo_fini
  -Wl,--wrap=rocket_batched_submit_supported -Wl,--wrap=co_execute)
add_test(NAME multitile_host COMMAND multitile_host_test)
set_tests_properties(multitile_host PROPERTIES ENVIRONMENT "ROCKET_CHIP=rk3588")
add_executable(multitile_probe tools/c/multitile_probe.c)
target_link_libraries(multitile_probe PRIVATE custom_ops)
target_compile_options(multitile_probe PRIVATE -Wall -Wextra -Werror)
add_executable(graph_host_test tests/c/graph_host_test.c)
target_link_libraries(graph_host_test PRIVATE custom_ops)
target_compile_options(graph_host_test PRIVATE -Wall -Wextra -Werror)
target_link_options(graph_host_test PRIVATE
  -Wl,--wrap=rocket_bo_alloc32 -Wl,--wrap=rocket_bo_free
  -Wl,--wrap=rocket_bo_prep -Wl,--wrap=rocket_bo_fini
  -Wl,--wrap=rocket_batched_submit_supported -Wl,--wrap=co_execute
  -Wl,--wrap=co_emit_ew -Wl,--wrap=co_emit_lut)
add_test(NAME graph_host COMMAND graph_host_test)
set_tests_properties(graph_host PROPERTIES ENVIRONMENT "ROCKET_CHIP=rk3588")
add_executable(graph_ffn tools/c/graph_cli.c)
target_link_libraries(graph_ffn PRIVATE custom_ops)
target_compile_options(graph_ffn PRIVATE -Wall -Wextra -Werror)
target_link_options(graph_ffn PRIVATE -Wl,--wrap=co_upload_f16 -Wl,--wrap=co_download)
add_test(NAME graph_cli_help COMMAND graph_ffn --help)
add_executable(row_graph tools/c/row_cli.c)
target_link_libraries(row_graph PRIVATE custom_ops)
target_compile_options(row_graph PRIVATE -Wall -Wextra -Werror)
target_link_options(row_graph PRIVATE -Wl,--wrap=co_upload_f16 -Wl,--wrap=co_download)
add_test(NAME row_cli_help COMMAND row_graph --help)
add_test(NAME row_campaign_artifacts COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/row_campaign_test.py")
add_executable(attention_probe tools/c/attention_probe.c)
target_link_libraries(attention_probe PRIVATE custom_ops)
target_compile_options(attention_probe PRIVATE -Wall -Wextra -Werror)
target_link_options(attention_probe PRIVATE -Wl,--wrap=co_upload_f16 -Wl,--wrap=co_download -Wl,--wrap=co_execute)
add_executable(decoder_probe tools/c/decoder_probe.c)
target_link_libraries(decoder_probe PRIVATE custom_ops)
target_compile_options(decoder_probe PRIVATE -Wall -Wextra -Werror)
target_link_options(decoder_probe PRIVATE -Wl,--wrap=co_upload_f16 -Wl,--wrap=co_download -Wl,--wrap=co_graph_run)

add_executable(regcmd_overlap_probe tools/c/regcmd_overlap_probe.c)
target_include_directories(regcmd_overlap_probe PRIVATE "${UPSTREAM_ROOT}/rknpu-submit/src")
target_link_libraries(regcmd_overlap_probe PRIVATE co_schedule)
target_compile_options(regcmd_overlap_probe PRIVATE -Wall -Wextra -Werror)
add_test(NAME regcmd_overlap_host COMMAND regcmd_overlap_probe host)
set_tests_properties(regcmd_overlap_host PROPERTIES ENVIRONMENT "ROCKET_CHIP=rk3588")
add_test(NAME matched_overlap_evidence COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/matched_overlap_analyze_test.py")

add_executable(gdn_probe tools/c/gdn_probe.c)
target_link_libraries(gdn_probe PRIVATE custom_ops)
target_compile_options(gdn_probe PRIVATE -Wall -Wextra -Werror)
add_test(NAME gdn_host COMMAND gdn_probe host)
set_tests_properties(gdn_host PROPERTIES ENVIRONMENT "ROCKET_CHIP=rk3588")
target_link_options(gdn_probe PRIVATE -Wl,--wrap=co_upload_f16 -Wl,--wrap=co_download -Wl,--wrap=co_execute)
add_test(NAME gdn_evidence_negative COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/gdn_gate_test.py")
add_executable(gdn_experiments tools/c/gdn_experiments.c)
target_link_libraries(gdn_experiments PRIVATE custom_ops)
target_compile_options(gdn_experiments PRIVATE -Wall -Wextra -Werror)
add_executable(gdn_cpu_test tests/c/gdn_cpu_test.c)
target_link_libraries(gdn_cpu_test PRIVATE custom_ops)
target_compile_options(gdn_cpu_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME gdn_cpu_stages COMMAND gdn_cpu_test)
add_test(NAME gdn_experiments_evidence COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/python/gdn_experiments_test.py")

# TileLang -> RKIR -> RKPlan descriptor ABI and synchronous runtime.
set_target_properties(custom_ops PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(rk_plan_core STATIC src/rk_plan_validate.c)
target_include_directories(rk_plan_core PUBLIC include PRIVATE src)
target_link_libraries(rk_plan_core PUBLIC custom_ops)
target_compile_options(rk_plan_core PRIVATE -Wall -Wextra -Werror)
set_target_properties(rk_plan_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(rk_plan_runtime_obj OBJECT src/rk_plan_runtime.c src/rk_plan_emit.c)
target_include_directories(rk_plan_runtime_obj PRIVATE include src)
target_link_libraries(rk_plan_runtime_obj PUBLIC custom_ops)
target_compile_options(rk_plan_runtime_obj PRIVATE -Wall -Wextra -Werror)
set_target_properties(rk_plan_runtime_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(rknpu_plan SHARED $<TARGET_OBJECTS:rk_plan_runtime_obj>
  src/rk_plan_target.c)
target_include_directories(rknpu_plan PUBLIC include PRIVATE src)
target_link_libraries(rknpu_plan PRIVATE rk_plan_core custom_ops)
set_target_properties(rknpu_plan PROPERTIES OUTPUT_NAME rknpu_plan)

# Identical runtime with the timing probes compiled out.  This is the
# end-to-end control for "does the envelope instrumentation cost anything": the
# same sources, the same ABI, only now_us()/the probe counter removed.  It is a
# measurement target, never the production library.
add_library(rk_plan_runtime_noprobe_obj OBJECT src/rk_plan_runtime.c
  src/rk_plan_emit.c)
target_include_directories(rk_plan_runtime_noprobe_obj PRIVATE include src)
target_link_libraries(rk_plan_runtime_noprobe_obj PUBLIC custom_ops)
target_compile_options(rk_plan_runtime_noprobe_obj PRIVATE -Wall -Wextra
  -Werror -DRK_PLAN_PROBE_OFF=1)
set_target_properties(rk_plan_runtime_noprobe_obj PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
add_library(rknpu_plan_noprobe SHARED $<TARGET_OBJECTS:rk_plan_runtime_noprobe_obj>
  src/rk_plan_target.c)
target_include_directories(rknpu_plan_noprobe PUBLIC include PRIVATE src)
target_link_libraries(rknpu_plan_noprobe PRIVATE rk_plan_core custom_ops)
set_target_properties(rknpu_plan_noprobe PROPERTIES OUTPUT_NAME rknpu_plan_noprobe)
# Test-only shared library: same runtime objects and C ABI, but the vendor calls
# resolve to the mock seam.  The production rknpu_plan target never links it.
add_library(rknpu_plan_mock SHARED $<TARGET_OBJECTS:rk_plan_runtime_obj>
  tests/mock/npu_mock.c)
target_include_directories(rknpu_plan_mock PRIVATE include src)
target_link_libraries(rknpu_plan_mock PRIVATE rk_plan_core m)
target_compile_options(rknpu_plan_mock PRIVATE -Wall -Wextra -Werror)
target_link_options(rknpu_plan_mock PRIVATE
  -Wl,--wrap=rocket_bo_alloc32 -Wl,--wrap=rocket_bo_free
  -Wl,--wrap=rocket_bo_prep -Wl,--wrap=rocket_bo_fini
  -Wl,--wrap=rocket_batched_submit_supported -Wl,--wrap=co_execute_profiled
  -Wl,--wrap=co_execute_cached_profiled
  -Wl,--wrap=co_execute_plan_profiled
  -Wl,--wrap=co_emit_lut -Wl,--wrap=co_emit_ew
  -Wl,--wrap=co_emit_ew_max
  -Wl,--wrap=co_emit_mm -Wl,--wrap=co_emit_mm_acc
  -Wl,--wrap=rocket_context_faulted)
set_target_properties(rknpu_plan_mock PROPERTIES OUTPUT_NAME rknpu_plan_mock)

# Compile-time check that the test-only mock API matches the ctypes table.
add_executable(mock_signature_probe tests/mock/mock_signature_probe.c
  tests/mock/npu_mock.c)
target_include_directories(mock_signature_probe PRIVATE include src tests/mock)
target_link_libraries(mock_signature_probe PRIVATE custom_ops m)
target_link_options(mock_signature_probe PRIVATE
  -Wl,--wrap=rocket_bo_alloc32 -Wl,--wrap=rocket_bo_free
  -Wl,--wrap=rocket_bo_prep -Wl,--wrap=rocket_bo_fini
  -Wl,--wrap=rocket_batched_submit_supported -Wl,--wrap=co_execute_profiled
  -Wl,--wrap=co_emit_lut -Wl,--wrap=co_emit_ew
  -Wl,--wrap=co_emit_ew_max
  -Wl,--wrap=co_emit_mm -Wl,--wrap=co_emit_mm_acc
  -Wl,--wrap=rocket_context_faulted)
target_compile_options(mock_signature_probe PRIVATE -Wall -Wextra -Werror)

# Test-only scenarios driven through the production emitters.
add_executable(rk_mock_scenarios tests/c/mock_scenarios.c tests/mock/npu_mock.c)
target_include_directories(rk_mock_scenarios PRIVATE include src tests/mock)
target_link_libraries(rk_mock_scenarios PRIVATE custom_ops m)
target_compile_options(rk_mock_scenarios PRIVATE -Wall -Wextra -Werror)
target_link_options(rk_mock_scenarios PRIVATE
  -Wl,--wrap=rocket_bo_alloc32 -Wl,--wrap=rocket_bo_free
  -Wl,--wrap=rocket_bo_prep -Wl,--wrap=rocket_bo_fini
  -Wl,--wrap=rocket_batched_submit_supported -Wl,--wrap=co_execute_profiled
  -Wl,--wrap=co_emit_lut -Wl,--wrap=co_emit_ew
  -Wl,--wrap=co_emit_ew_max
  -Wl,--wrap=co_emit_mm -Wl,--wrap=co_emit_mm_acc
  -Wl,--wrap=rocket_context_faulted)

# Exercise execute_impl's production submit/output-sync loop.  The submit stub
# only substitutes the device ioctl; command construction, recursion and BO
# prep/fini sequencing remain the real custom_ops.c implementation.
add_executable(output_sync_test tests/c/output_sync_test.c tests/mock/npu_mock.c)
target_include_directories(output_sync_test PRIVATE include src tests/mock)
target_link_libraries(output_sync_test PRIVATE custom_ops m)
target_compile_options(output_sync_test PRIVATE -Wall -Wextra -Werror)
target_link_options(output_sync_test PRIVATE
  -Wl,--wrap=rocket_bo_alloc32 -Wl,--wrap=rocket_bo_free
  -Wl,--wrap=rocket_bo_prep -Wl,--wrap=rocket_bo_fini
  -Wl,--wrap=rocket_batched_submit_supported
  -Wl,--wrap=rocket_submit_tasks_flags
  -Wl,--wrap=co_emit_lut -Wl,--wrap=co_emit_ew -Wl,--wrap=co_emit_ew_max
  -Wl,--wrap=co_emit_mm -Wl,--wrap=co_emit_mm_acc
  -Wl,--wrap=rocket_context_faulted)
add_test(NAME output_sync COMMAND output_sync_test)

# Command-BO packing reuse: real clear/pack/seal/verify/handle-list path, with
# only the submit ioctl stubbed.  Proves cached and per-submit plans produce the
# same bytes/descriptors/handles and that the opt-in command sync is separable.
add_executable(command_cache_test tests/c/command_cache_test.c tests/mock/npu_mock.c)
target_include_directories(command_cache_test PRIVATE include src tests/mock)
target_link_libraries(command_cache_test PRIVATE custom_ops m)
target_compile_options(command_cache_test PRIVATE -Wall -Wextra -Werror)
target_link_options(command_cache_test PRIVATE
  -Wl,--wrap=rocket_bo_alloc32 -Wl,--wrap=rocket_bo_free
  -Wl,--wrap=rocket_bo_prep -Wl,--wrap=rocket_bo_fini
  -Wl,--wrap=rocket_batched_submit_supported
  -Wl,--wrap=rocket_submit_tasks_flags
  -Wl,--wrap=co_emit_lut -Wl,--wrap=co_emit_ew -Wl,--wrap=co_emit_ew_max
  -Wl,--wrap=co_emit_mm -Wl,--wrap=co_emit_mm_acc
  -Wl,--wrap=rocket_context_faulted)
add_test(NAME command_cache COMMAND command_cache_test)
set_tests_properties(command_cache PROPERTIES ENVIRONMENT "ROCKET_CHIP=rk3588")

# Guard-scan coverage: the vectorised scan must keep byte-exact coverage.
add_executable(guard_scan_test tests/c/guard_scan_test.c tests/mock/npu_mock.c)
target_include_directories(guard_scan_test PRIVATE include src tests/mock)
target_link_libraries(guard_scan_test PRIVATE custom_ops m)
target_compile_options(guard_scan_test PRIVATE -Wall -Wextra -Werror)
target_link_options(guard_scan_test PRIVATE
  -Wl,--wrap=rocket_bo_alloc32 -Wl,--wrap=rocket_bo_free
  -Wl,--wrap=rocket_bo_prep -Wl,--wrap=rocket_bo_fini
  -Wl,--wrap=rocket_batched_submit_supported -Wl,--wrap=co_execute_profiled
  -Wl,--wrap=co_emit_lut -Wl,--wrap=co_emit_ew
  -Wl,--wrap=co_emit_ew_max
  -Wl,--wrap=co_emit_mm -Wl,--wrap=co_emit_mm_acc
  -Wl,--wrap=rocket_context_faulted)
add_test(NAME guard_scan COMMAND guard_scan_test)

# CUBE-F16 -> row-major fast download: exact logical order, no padding copied.
add_executable(download_fastpath_test tests/c/download_fastpath_test.c
  tests/mock/npu_mock.c)
target_include_directories(download_fastpath_test PRIVATE include src tests/mock)
target_link_libraries(download_fastpath_test PRIVATE custom_ops m)
target_compile_options(download_fastpath_test PRIVATE -Wall -Wextra -Werror)
target_link_options(download_fastpath_test PRIVATE
  -Wl,--wrap=rocket_bo_alloc32 -Wl,--wrap=rocket_bo_free
  -Wl,--wrap=rocket_bo_prep -Wl,--wrap=rocket_bo_fini
  -Wl,--wrap=rocket_batched_submit_supported -Wl,--wrap=co_execute_profiled
  -Wl,--wrap=co_emit_lut -Wl,--wrap=co_emit_ew
  -Wl,--wrap=co_emit_ew_max
  -Wl,--wrap=co_emit_mm -Wl,--wrap=co_emit_mm_acc
  -Wl,--wrap=rocket_context_faulted)
add_test(NAME download_fastpath COMMAND download_fastpath_test)

# Download helpers: exhaustive FP16 finiteness and publish-rectangle equivalence.
add_executable(fp16_scan_test tests/c/fp16_scan_test.c)
target_include_directories(fp16_scan_test PRIVATE src)
target_link_libraries(fp16_scan_test PRIVATE m)
target_compile_options(fp16_scan_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME fp16_scan COMMAND fp16_scan_test)

# Per-task reference comparison for the B2 chain probe (host-checkable).
add_executable(task_ref_test tests/c/task_ref_test.c)
target_include_directories(task_ref_test PRIVATE src)
target_compile_options(task_ref_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME task_ref COMMAND task_ref_test)

add_executable(rk_plan_contract_test tests/c/rk_plan_contract_test.c)
target_link_libraries(rk_plan_contract_test PRIVATE rk_plan_core custom_ops)
target_compile_options(rk_plan_contract_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME rk_plan_contract COMMAND rk_plan_contract_test)
add_executable(rk_plan_host_test tests/c/rk_plan_host_test.c
  tests/mock/npu_mock.c $<TARGET_OBJECTS:rk_plan_runtime_obj>)
target_include_directories(rk_plan_host_test PRIVATE include src tests/mock)
target_link_libraries(rk_plan_host_test PRIVATE rk_plan_core custom_ops m)
target_compile_options(rk_plan_host_test PRIVATE -Wall -Wextra -Werror)
target_link_options(rk_plan_host_test PRIVATE
  -Wl,--wrap=rocket_bo_alloc32 -Wl,--wrap=rocket_bo_free
  -Wl,--wrap=rocket_bo_prep -Wl,--wrap=rocket_bo_fini
  -Wl,--wrap=rocket_batched_submit_supported -Wl,--wrap=co_execute_profiled
  -Wl,--wrap=co_execute_cached_profiled
  -Wl,--wrap=co_execute_plan_profiled
  -Wl,--wrap=co_emit_lut -Wl,--wrap=co_emit_ew
  -Wl,--wrap=co_emit_ew_max
  -Wl,--wrap=co_emit_mm -Wl,--wrap=co_emit_mm_acc
  -Wl,--wrap=rocket_context_faulted)
add_test(NAME rk_plan_host COMMAND rk_plan_host_test)

# Provider-boundary quarantine test: includes the real provider source, seeds a
# fake per-fd context, and proves a queued submit cannot rewrite the task BO after
# another submit fails while holding the provider submit lock.
add_executable(provider_context_test
  "${UPSTREAM_ROOT}/rknpu-submit/tests/provider_context_test.c")
target_include_directories(provider_context_test PRIVATE
  ${RKT}/include ${RKT}/src "${UPSTREAM_ROOT}/rknpu-submit/include"
  "${UPSTREAM_ROOT}/rknpu-submit/src")
find_package(Threads REQUIRED)
target_link_libraries(provider_context_test PRIVATE Threads::Threads m)
target_compile_options(provider_context_test PRIVATE -Wall -Wextra -Werror)
add_test(NAME provider_context COMMAND provider_context_test)
# MALLOC_PERTURB_ makes any read of uninitialised Host staging memory
# deterministic instead of silently seeing zeroed fresh heap pages.
set_tests_properties(rk_plan_host PROPERTIES
  ENVIRONMENT "ROCKET_CHIP=rk3588;MALLOC_PERTURB_=128")

# Primitive cost floor for the plan envelope.  This is a measurement tool, not a
# runtime path and deliberately not a ctest case: without a device it reports
# "unsupported" and exits 77, while on a board it succeeds, so its exit status is
# environment-dependent.  Campaign scripts record its JSON lines instead.
add_executable(plan_envelope_probe tools/c/plan_envelope_probe.c)
target_include_directories(plan_envelope_probe PRIVATE "${RKT}/include")
target_link_libraries(plan_envelope_probe PRIVATE custom_ops)
target_compile_options(plan_envelope_probe PRIVATE -Wall -Wextra -Werror)

# Bounded B2 validity experiment: N independent matmuls in one chain vs N separate
# submits, with per-task input/weight/output BOs and a bitwise comparison.  A
# measurement tool only; no recipe, plan or runtime path is added by it.
add_executable(independent_mm_chain_probe tools/c/independent_mm_chain_probe.c)
target_include_directories(independent_mm_chain_probe PRIVATE "${RKT}/include" src)
target_link_libraries(independent_mm_chain_probe PRIVATE custom_ops)
target_compile_options(independent_mm_chain_probe PRIVATE -Wall -Wextra -Werror)
