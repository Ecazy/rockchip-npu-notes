#!/usr/bin/env python3
"""Offline artifact integrity, measured dataset structure, archived trace replay."""
import argparse
import json
from pathlib import Path
from dataset import ROOT, HERE, digest, validate
from replay import replay

def verify(root=ROOT, full_sources=False):
    root = Path(root).resolve()
    module = root/'research/multidevice-scheduling'
    checked = 0
    def check(row, path_field='path'):
        nonlocal checked
        path = (root/row[path_field]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != row['sha256']:
            raise ValueError(f'missing/changed artifact {row[path_field]}')
        checked += 1
    for row in json.loads((module/'static-analysis/source-manifest.json').read_text())['sources']:
        check(row,'local_path')
    for row in json.loads((module/'static-analysis/binary-manifest.json').read_text())['reports']:
        check(row)
    archive = module/'evidence/rk3588-trace-20261010'
    for row in json.loads((archive/'archive-manifest.json').read_text())['files']:
        check(row,'archived_path')
    doc = json.loads((module/'data/job-costs.json').read_text())
    manifest = json.loads((module/'data/source-manifest.json').read_text())
    stats = validate(doc,manifest,root,full_sources)
    replays = []
    for block in sorted(archive.glob('block-00-cell-*')):
        result = replay(block)
        replays.append({'block':block.name, **result['summary']})
    hardware_reports=[]
    for folder in sorted((module/'results').glob('backend-*')):
        report = json.loads((folder/'smoke.json').read_text())
        if (folder/'artifact-manifest.json').exists():
            for row in json.loads((folder/'artifact-manifest.json').read_text())['files']:
                check(row)
        if report.get('status')!='PASS' or (folder/'exit-code.txt').read_text().strip()!='0':
            raise ValueError('hardware smoke not PASS')
        hardware_reports.append(folder.name)
    return {'status':'PASS', 'artifact_files_verified':checked, 'dataset':stats,
            'archived_trace_replays':replays, 'hardware_smoke':hardware_reports,
            'source_scope':'full original sources verified' if full_sources else
              'dataset structure + archived traces + static snapshots; other historical sources not checked'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--full-sources',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path)
    args=p.parse_args()
    try: report=verify(args.root,args.full_sources)
    except (ValueError,OSError,KeyError) as e:
        print(json.dumps({'status':'FAIL','reason':str(e)}));return 1
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'artifacts':report['artifact_files_verified'],
                      'dataset':report['dataset'],'trace_requests':sum(r['requests'] for r in report['archived_trace_replays']),
                      'hardware_smoke':report['hardware_smoke']},sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
