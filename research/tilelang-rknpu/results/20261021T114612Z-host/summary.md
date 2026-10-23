# Host verification 20261021T114612Z-host

overall exit: **0**
evidence complete: **True**

| step | exit |
| --- | --- |
| environment | 0 |
| isolation | 0 |
| source-hashes | 0 |
| prepare-build | 0 |
| configure | 0 |
| build | 0 |
| ctest | 0 |
| rk_plan_contract | 0 |
| rk_plan_host | 0 |
| configure-sanitize | 0 |
| build-sanitize | 0 |
| sanitize-contract | 0 |
| sanitize-host | 0 |
| archive-flags | 0 |
| pytest | 0 |

ctest: {'passed_percent': 100, 'failed': 0, 'total': 26}
C checks: contract={'checks': 26, 'failures': 0} host={'checks': 67, 'failures': 0}
sanitizer: {'enabled_explicitly': True, 'flags': '-fsanitize=address,undefined'}
pytest: 60 passed, 0 skipped

Device access: none. Board acceptance: not run. This record proves Host behaviour only; it is not an RK3588 acceptance report.
