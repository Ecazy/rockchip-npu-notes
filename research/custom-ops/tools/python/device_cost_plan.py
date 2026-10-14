#!/usr/bin/env python3
"""Emit the exact v2 matrix consumed by device_cost_campaign."""
import argparse,json,pathlib
from device_cost_campaign import matrix
from device_cost_schema import enumerate_config
def build(suite,blocks=5,operations=600,warmup=20):
    cells=matrix(suite)
    for c in cells:c.update(operations=operations,warmup=warmup,seed=20261010,timeout_s=120)
    return dict(schema_version=2,suite=suite,blocks=blocks,cells=cells)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',choices=('smoke','cost','concurrency','resident','resident-smoke','saturation','interference','trace','tail'),default='smoke')
    p.add_argument('--blocks',type=int,default=5);p.add_argument('--operations',type=int,default=600);p.add_argument('--warmup',type=int,default=20)
    p.add_argument('--output',type=pathlib.Path);a=p.parse_args();c=build(a.suite,a.blocks,a.operations,a.warmup)
    if a.output:a.output.write_text(json.dumps(c,indent=2)+'\n')
    print(json.dumps(enumerate_config(c),indent=2))
