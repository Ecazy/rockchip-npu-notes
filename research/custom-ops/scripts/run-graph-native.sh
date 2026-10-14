#!/usr/bin/env bash
set -euo pipefail
gc_project=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
gc_build=${1:-$gc_project/build-graph}
gc_results=${2:-$gc_project/artifacts/graph-$(date +%Y%m%d-%H%M%S)}
[[ ! -e "$gc_results" ]] || { echo 'Use a fresh results directory' >&2; exit 1; }
mkdir -p "$gc_results"
exec 9>/tmp/rk3588-custom-ops.lock
flock -n 9 || { echo 'Another NPU campaign holds the lock' >&2; exit 1; }
gc_upstream=${UPSTREAM_ROOT:-$gc_project/../upstream-components}
python3 "$gc_project/tools/python/summarize.py" --sources "$gc_project" "$gc_upstream" > "$gc_results/source-manifest.json"
{ date -Is; uname -a; } > "$gc_results/environment.log"
journalctl -k -n 0 --show-cursor --no-pager > "$gc_results/kernel-before.log" 2>&1 || true
gc_cursor=$(sed -n 's/^-- cursor: //p' "$gc_results/kernel-before.log")
gc_capture_kernel() {
    if [[ -n "$gc_cursor" ]]; then
        journalctl -k --after-cursor="$gc_cursor" --no-pager > "$gc_results/kernel-interval.log" 2>&1 || true
    fi
}
trap gc_capture_kernel EXIT
cmake -S "$gc_project" -B "$gc_build" -DUPSTREAM_ROOT="$gc_upstream" -DCMAKE_BUILD_TYPE=Release > "$gc_results/configure.log" 2>&1
cmake --build "$gc_build" -j4 > "$gc_results/build.log" 2>&1
ctest --test-dir "$gc_build" --output-on-failure > "$gc_results/host-tests.log" 2>&1
export ROCKET_CHIP=rk3588
for gc_mask in 1 2 4; do
    RKNPU_CORE_MASK=$gc_mask python3 "$gc_project/tools/python/graph_gate.py" "$gc_build/graph_ffn" "$gc_results/core-$gc_mask-graph" > "$gc_results/core-$gc_mask-graph.log" 2>&1
    RKNPU_CORE_MASK=$gc_mask "$gc_build/multitile_probe" > "$gc_results/core-$gc_mask-device-regression.log" 2>&1
    RKNPU_CORE_MASK=$gc_mask python3 "$gc_project/tools/python/cli_gate.py" "$gc_build/linear_hardsigmoid" "$gc_results/core-$gc_mask-cli-regression" > "$gc_results/core-$gc_mask-cli-regression.log" 2>&1
done
gc_capture_kernel
python3 - "$gc_results" <<'PY'
import json, re, sys
from pathlib import Path
root=Path(sys.argv[1])
checks=[]
for mask in (1,2,4):
    graph=json.loads((root/f'core-{mask}-graph/graph-report.json').read_text())
    cli=json.loads((root/f'core-{mask}-cli-regression/cli-report.json').read_text())
    assert graph['status']=='PASS' and len(graph['checks'])==16
    assert cli['status']=='PASS' and len(cli['checks'])==25
    log=(root/f'core-{mask}-device-regression.log').read_text()
    assert len(re.findall(r'^MULTITILE .* PASS$',log,re.M))==8
    checks.append({'core_mask':mask,'graph_checks':16,'graph_negative_checks':2,
        'max_abs':max(c['max_abs'] for c in graph['checks']),
        'cli_regression_checks':25,'device_regression_cases':8})
interval=root/'kernel-interval.log'
if interval.exists():
    assert not re.search(r'rknpu.*(?:timeout|fault)|iommu.*fault|Unhandled fault',interval.read_text(),re.I)
(root/'graph-summary.json').write_text(json.dumps({'status':'PASS','checks':checks,
    'kernel_interval_available':interval.exists()},indent=2)+'\n')
PY
printf 'GRAPH hardware campaign PASS\n'
