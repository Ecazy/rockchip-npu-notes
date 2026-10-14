#!/usr/bin/env python3
"""Recompute decoder block/cache witnesses and summarize paired API timings."""
import argparse
import json
from pathlib import Path
import re
from decoder_gate import verify
from attention_gate import verify as attention_verify


def run(root):
    if '100% tests passed' not in (root/'host-tests.log').read_text():raise ValueError('host test failure')
    checks=[]
    for core in (1,2,4):
        for mode in ('chain','separate'):
            folder=root/f'core-{core}-{mode}';stored=json.loads((folder/'decoder-report.json').read_text())
            if verify(folder,mode)!=stored:raise ValueError('decoder witness/hash/oracle differs')
            checks.append({'core_mask':core,'mode':mode,'cases':3,
                'block_max_abs':max(max(c['baseline_max_abs'],c['optimized_max_abs'],c['full_max_abs']) for c in stored['checks']),
                'cache_max_abs':max(c['cache_max_abs'] for c in stored['checks']),
                'geometry_max_abs':max(c['optimized_vs_baseline_max_abs'] for c in stored['checks']),
                'post_ffn_rollback_cases':3,
                'single_token_api_ratios':[c['single_token_api_ratio'] for c in stored['checks']]})
        for ident in range(3):
            for field in ('baseline-y.bin','optimized-y.bin','full-y.bin','baseline-k.bin','baseline-v.bin','optimized-k.bin','optimized-v.bin'):
                if (root/f'core-{core}-chain/case-{ident}-{field}').read_bytes()!=(root/f'core-{core}-separate/case-{ident}-{field}').read_bytes():
                    raise ValueError('chain/separate block/cache byte mismatch')
        regression=root/f'core-{core}-attention-regression'
        if attention_verify(regression,'chain')!=json.loads((regression/'attention-report.json').read_text()):raise ValueError('attention regression differs')
        if len(re.findall(r'^MULTITILE .* PASS$',(root/f'core-{core}-device-regression.log').read_text(),re.M))!=8:raise ValueError('device regression incomplete')
    interval=root/'kernel-interval.log'
    if interval.exists() and re.search(r'rknpu.*(?:timeout|fault)|iommu.*fault|Unhandled fault',interval.read_text(),re.I):raise ValueError('kernel fault/timeout')
    result={'status':'PASS','checks':checks,'kernel_interval_available':interval.exists(),'oracles_recomputed':True,
            'timing_scope':'paired alternating warm API calls; 3 repeats per schedule; not model tokens/sec or a production benchmark'}
    target=root/'decoder-summary.json'
    if target.exists():
        if json.loads(target.read_text())!=result:raise ValueError('existing summary differs')
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('results',type=Path);args=parser.parse_args();print(json.dumps(run(args.results),indent=2))
