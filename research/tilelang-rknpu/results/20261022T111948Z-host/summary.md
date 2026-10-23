# Host verification 20261022T111948Z-host

overall exit: **1**
evidence complete: **False**

| step | exit |
| --- | --- |
| environment | 0 |
| isolation | 0 |
| source-hashes | 0 |
| prepare-build | 0 |
| configure | 0 |
| build | 0 |
| ctest | 8 |
| rk_plan_contract | 0 |
| rk_plan_host | 1 |
| graph_host | 0 |
| configure-sanitize | 0 |
| build-sanitize | 0 |
| sanitize-contract | 0 |
| sanitize-host | 1 |
| archive-flags | 0 |
| pytest | 0 |

ctest: {'passed_percent': 96, 'failed': 1, 'total': 26}
C checks: contract={'checks': 26, 'failures': 0} host={'checks': 688, 'failures': 29}
sanitizer: {'enabled_explicitly': True, 'flags': '-fsanitize=address,undefined'}
pytest: 130 passed, 0 failed, 0 skipped

## Evidence problems

- rk_plan_host.log reported 29 failures

Device access: none. Board acceptance: not run. This record proves Host behaviour only; it is not an RK3588 acceptance report.
