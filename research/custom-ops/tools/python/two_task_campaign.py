#!/usr/bin/env python3
"""Campaign coverage, integrity and kernel gate (all 12 process configurations)."""
import hashlib
import json
from pathlib import Path
import re
import sys
from two_task_verify import verify_process


def verify(root, rounds):
    errors=[];runs=[]
    for mask in (1,2,4):
        for sptr in (14,48):
            for rep in (1,2):
                name=f'core-{mask}-sptr-{sptr}-run-{rep}';p=root/name
                result=verify_process(p)
                if result['status']!='PASS':errors.append(f'{name}: {result["errors"]}')
                elif (result['stats']['rounds'],result['stats']['core_mask'],result['stats']['sptr'])!=(rounds,mask,sptr):
                    errors.append(f'{name}: coverage/configuration mismatch')
                try:
                    if int((p/'exit.txt').read_text())!=0:errors.append(f'{name}: process failed')
                except (OSError,ValueError):errors.append(f'{name}: missing exit status')
                if (p/'validation.json').exists():
                    saved=json.loads((p/'validation.json').read_text())
                    if saved!=result:errors.append(f'{name}: recorded evidence hashes or validation changed')
                else:errors.append(f'{name}: missing saved validation')
                runs.append(dict(name=name,**result))
    kernel=root/'kernel-after.log'
    if not kernel.exists():errors.append('missing kernel log')
    else:
        faults=[s for s in kernel.read_text().splitlines() if re.search(r'(rknpu|npu|iommu).*(timeout|timed out|fault|reset|failed)',s,re.I)]
        if faults:errors.extend(faults)
    host=root/'host-tests.log'
    if not host.exists() or '100% tests passed' not in host.read_text():errors.append('host tests did not pass')
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir())
            if p.is_file() and p.name!='campaign.json'}
    result=dict(status='FAIL' if errors else 'PASS',errors=errors,runs=runs,sha256=hashes,
                rounds=sum(r['stats'].get('rounds',0) for r in runs),
                checks=sum(r['stats'].get('checks',0) for r in runs),
                submits=sum(r['stats'].get('submits',0) for r in runs),
                tasks=sum(r['stats'].get('tasks',0) for r in runs))
    saved=root/'campaign.json'
    if saved.exists():
        if json.loads(saved.read_text())!=result:
            print('FAIL campaign evidence changed; original manifest preserved');return 1
    else:saved.write_text(json.dumps(result,indent=2)+'\n')
    print(f"TWO_TASK_CAMPAIGN {result['status']} processes={len(runs)} rounds={result['rounds']} checks={result['checks']} submits={result['submits']} tasks={result['tasks']}")
    for error in errors:print(error)
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    if len(sys.argv)!=3:sys.exit('usage: two_task_campaign.py RESULTS ROUNDS')
    sys.exit(verify(Path(sys.argv[1]),int(sys.argv[2])))
