#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Summarize raw native probe A/B rounds without dropping failures."""
import argparse, json, statistics
from pathlib import Path


def parse(path):
    text = path.read_text()
    for line in text.splitlines():
        if line.startswith("WARM_US"):
            return [float(x) for x in line.split()[1:]]
    raise ValueError(f"missing WARM_US: {path}")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--experiment',default='finite-scan')
    a=ap.parse_args(); cases={}; failures=[]
    for case in ('qk_b8','pv_b8'):
        rounds=[]
        for ri in range(1,8):
            vals={}
            for arm in ('fast','reference'):
                base=a.root/case/f'r{ri}'/arm
                try:
                    rc=int(Path(str(base)+'.exit').read_text().strip())
                    if rc: raise RuntimeError(f'exit={rc}')
                    vals[arm]=parse(Path(str(base)+'.stdout'))
                    if not vals[arm]: raise ValueError('empty samples')
                except Exception as e: failures.append(f'{case} r{ri} {arm}: {e}')
            if len(vals)==2:
                try:
                    same=(Path(str(a.root/case/f'r{ri}'/'fast.bin')).read_bytes()==
                          Path(str(a.root/case/f'r{ri}'/'reference.bin')).read_bytes())
                except Exception as e:
                    failures.append(f'{case} r{ri}: output file unavailable: {e}')
                    same=False
                med={k:statistics.median(v) for k,v in vals.items()}
                rounds.append({'round':ri,'fast_median_us':med['fast'],
                    'reference_median_us':med['reference'],
                    'ratio_reference_over_fast':med['reference']/med['fast'],
                    'bitwise':same,'fast_n':len(vals['fast']),'reference_n':len(vals['reference'])})
                if not same: failures.append(f'{case} r{ri}: output bytes differ')
        ratios=[r['ratio_reference_over_fast'] for r in rounds]
        cases[case]={'rounds':rounds,'fast_median_us':statistics.median([r['fast_median_us'] for r in rounds]) if rounds else None,
            'reference_median_us':statistics.median([r['reference_median_us'] for r in rounds]) if rounds else None,
            'ratio_reference_over_fast_median':statistics.median(ratios) if ratios else None,
            'ratio_min':min(ratios) if ratios else None,'ratio_max':max(ratios) if ratios else None,
            'ratio_p10_aux':sorted(ratios)[(len(ratios)-1)//10] if ratios else None,
            'stable_positive_gain':bool(ratios) and min(ratios)>1.0,
            'all_bitwise':all(r['bitwise'] for r in rounds) and len(rounds)==7}
        if len(rounds)!=7: failures.append(f'{case}: only {len(rounds)}/7 valid rounds')
    report={'kind':a.experiment+'-isolated-summary','experiment':a.experiment,'primary':'median of per-round medians and corresponding ratios','p10_role':'auxiliary only','cases':cases,'failures':failures,'status':'PASS' if not failures else 'FAIL'}
    a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0 if not failures else 1
if __name__=='__main__': raise SystemExit(main())
