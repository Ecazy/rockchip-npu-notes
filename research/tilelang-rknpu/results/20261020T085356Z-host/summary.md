# Host verification 20261020T085356Z-host

overall exit: **1**
evidence complete: **False**

| step | exit |
| --- | --- |
| environment | 0 |
| source-hashes | 0 |
| prepare-build | 0 |
| configure | 1 |
| build | 1 |
| ctest | 1 |
| rk_plan_contract | 127 |
| rk_plan_host | 127 |
| configure-sanitize | 1 |
| build-sanitize | 1 |
| sanitize-contract | 127 |
| sanitize-host | 127 |
| archive-flags | 1 |
| pytest | 1 |

ctest: None
C checks: contract={'checks': None, 'failures': None} host={'checks': None, 'failures': None}
sanitizer: {'enabled_explicitly': True, 'flags': '-fsanitize=address,undefined'}
pytest: 58 passed, 1 skipped (SKIPPED [1] tests/test_ffi.py:26: rknpu_plan shared library not built: RK_E_UNSUPPORTED: cannot load rknpu_plan shared library: /home/maoyu/Documents/code/rockchip-npu-notes/research/custom-ops/build-host-mvp/librknpu_plan.so: cannot open shared object file: No such file or directory)

## Evidence problems

- missing or empty artifact: compile-flags/compile-commands-sanitize.json
- could not parse the ctest summary line
- could not parse check counts from rk_plan_contract.log
- could not parse check counts from rk_plan_host.log

Device access: none. Board acceptance: not run. This record proves Host behaviour only; it is not an RK3588 acceptance report.
