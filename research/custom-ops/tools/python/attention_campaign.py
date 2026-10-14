#!/usr/bin/env python3
"""Recompute every SDPA/RoPE witness and verify the immutable six-arm campaign."""
import argparse
import json
from pathlib import Path
import re
from attention_gate import verify


def run(root):
    if '100% tests passed' not in (root/'host-tests.log').read_text():raise ValueError('host tests incomplete')
    checks=[]
    for core in (1,2,4):
        for mode in ('chain','separate'):
            folder=root/f'core-{core}-{mode}'
            stored=json.loads((folder/'attention-report.json').read_text())
            recomputed=verify(folder,mode)
            if recomputed!=stored:raise ValueError('witness/hash/oracle changed')
            checks.append({'core_mask':core,'mode':mode,'cases':6,'rollback_cases':6,
                'stream_max_abs':max(c['stream_max_abs'] for c in stored['checks']),
                'graph_max_abs':max(c['graph_max_abs'] for c in stored['checks']),
                'cache_rope_max_abs':max(c['cache_rope_max_abs'] for c in stored['checks']),
                'cross_tile_rope_max_abs':stored['cross_tile_rope_max_abs']})
        for ident in (0,1,2,3,4,5):
            for name in ('stream-y.bin','graph-y.bin','cache-k.bin','cache-v.bin'):
                if (root/f'core-{core}-chain/case-{ident}-{name}').read_bytes()!=(root/f'core-{core}-separate/case-{ident}-{name}').read_bytes():
                    raise ValueError('chain/separate output/cache byte mismatch')
        for ident in (98,99):
            if (root/f'core-{core}-chain/case-{ident}-rope-y.bin').read_bytes()!=(root/f'core-{core}-separate/case-{ident}-rope-y.bin').read_bytes():
                raise ValueError('chain/separate RoPE byte mismatch')
        log=(root/f'core-{core}-device-regression.log').read_text()
        if len(re.findall(r'^MULTITILE .* PASS$',log,re.M))!=8:raise ValueError('device regression incomplete')
    interval=root/'kernel-interval.log'
    if interval.exists() and re.search(r'rknpu.*(?:timeout|fault)|iommu.*fault|Unhandled fault',interval.read_text(),re.I):
        raise ValueError('kernel fault/timeout')
    result={'status':'PASS','checks':checks,'kernel_interval_available':interval.exists(),'oracles_recomputed':True}
    target=root/'attention-summary.json'
    if target.exists():
        if json.loads(target.read_text())!=result:raise ValueError('existing summary differs')
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('results',type=Path);args=parser.parse_args()
    print(json.dumps(run(args.results),indent=2))
