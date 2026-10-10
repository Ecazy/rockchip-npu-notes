#!/usr/bin/env python3
"""Explicit real RK3588 smoke: zero FP16 32x32 -> HardSigmoid = 0.5."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backends.rk3588 import RK3588Backend
from backends.interface import DeviceUnavailable, UnsupportedOperation, TokenState

def main():
    try:
        with RK3588Backend() as backend:
            x = backend.alloc_buffer(2048, 'input')
            y = backend.alloc_buffer(2048, 'output')
            for b in (x, y):
                backend.configure_tensor(b, 'matrix', rows=32, cols=32, type_=0)
            backend.transfer(x, direction='upload', data=bytes(2048))
            backend.transfer(y, direction='upload', data=b'\x00\x7e'*1024)
            job = backend.create_lut_job(x, y)
            token = backend.submit(job)
            assert backend.wait(token) is TokenState.COMPLETE
            output = backend.transfer(y, direction='download', length=2048)
            if output != b'\x00\x38'*1024:
                raise RuntimeError('HardSigmoid output differs from exact 0.5')
            backend.check_guard(x); backend.check_guard(y)
            print(json.dumps({'status':'PASS', 'identity':backend.identity.as_dict(),
                              'elements':1024, 'task_count':1, 'expected':0.5,
                              'guard':'PASS', 'memory':backend.query_memory().__dict__}, sort_keys=True))
            backend.release_token(token); backend.release_job(job)
            backend.free_buffer(x); backend.free_buffer(y)
    except (DeviceUnavailable, UnsupportedOperation) as e:
        print(json.dumps({'status':'UNAVAILABLE','reason':str(e)}));return 77
    return 0
if __name__=='__main__':sys.exit(main())
