#!/usr/bin/env python3
"""Pinned boot-code fingerprint for the enqueue bound used by this experiment."""
import hashlib,json,pathlib,platform,subprocess,sys
RELEASE='6.1.43-rockchip-rk3588-rknpu098'
DIGEST='d186b9e6cb1e9935ba28e8284975ed6e4437065e1ea590cbaacfbc936e5a883d'
def audit(out):
    out=pathlib.Path(out)
    if platform.release()!=RELEASE:raise RuntimeError('kernel has not been audited for this timestamp contract')
    driver=pathlib.Path('/sys/kernel/debug/rknpu/version').read_text().strip()
    if driver!='RKNPU driver: v0.9.8':raise RuntimeError('wrong driver')
    mapping={line.split()[2]:int(line.split()[0],16) for line in pathlib.Path('/boot/System.map-'+RELEASE).read_text().splitlines() if len(line.split())==3}
    start=mapping['rknpu_job_schedule'];offset=start-mapping['_text']
    with open('/boot/vmlinuz-'+RELEASE,'rb') as image:image.seek(offset);code=image.read(1024)
    digest=hashlib.sha256(code).hexdigest()
    if digest!=DIGEST:raise RuntimeError('compiled queue ordering differs; redo disassembly/source audit')
    (out/'rknpu-job-schedule.bin').write_bytes(code)
    (out/'rknpu-job-schedule.disasm').write_text(subprocess.check_output(['objdump','-D','-b','binary','-m','aarch64',f'--adjust-vma={start}',str(out/'rknpu-job-schedule.bin')],text=True))
    result=dict(kernel=RELEASE,driver=driver,freq=pathlib.Path('/sys/kernel/debug/rknpu/freq').read_text().strip(),schedule_address=hex(start),get_task_number_address=hex(mapping['rknpu_get_task_number']),image_offset=offset,code_sha256=digest)
    (out/'kernel-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':audit(sys.argv[1])
