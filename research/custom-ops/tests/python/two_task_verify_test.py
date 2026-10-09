#!/usr/bin/env python3
"""Independent protocol fixtures and corruption tests; no NPU required."""
import sys
from pathlib import Path as _Path
sys.path.insert(0,str(_Path(__file__).resolve().parents[2]/"tools/python"))

import json
from pathlib import Path
import struct
import tempfile
import unittest

from two_task_verify import hsig, verify_process


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def fixture(path, fixed=False):
    events = []
    probe = []
    def emit(kind, **fields):
        events.append(dict(event_id=len(events)+1, event=kind, **fields))
    def create(i, size=32768):
        emit('create', id=i, fd=4, handle=i, obj=i+1000, dma=i*0x10000, size=size, domain=0)
        emit('map', id=i, fd=4, offset=i*4096)
        emit('mmap', id=i, address=i*0x100000, length=size)
    def destroy(i, size=32768):
        emit('unmap', id=i, address=i*0x100000, length=size)
        emit('destroy', id=i, fd=4, obj=i+1000, result=0)
    def vector(filename, values):
        path.joinpath(filename).write_bytes(struct.pack('<'+'e'*len(values), *values))
    for i in range(1,8):
        create(i)
    seq=0
    for round, (rows, cols) in enumerate(((16,32), (29,45))):
        data_bytes=(((max(rows,12)+3)//4)*4)*(((cols+31)//32)*32)*2
        for arm in 'ABCD':
            emit('stage', round=round, arm=arm)
            for ordinal in range(2 if arm=='B' else 1):
                seq+=1;rcid=7+seq;create(rcid)
                count=1 if arm in 'AB' else 2
                emit('submit_begin', submit_id=seq, fd=4, round=round, arm=arm, flags=1,
                     count=count, task_obj=1007, task_id=7, task_offset=0,
                     task_bytes=count*40, task_size=40, task_start=0, core_mask=1, domain=0, base=0)
                emit('subcore', submit_id=seq, index=0, start=0, count=count)
                for index in range(count):
                    srcid=1 if index==0 else 2
                    dstid=3 if index==0 else 4
                    if arm=='B' and ordinal:
                        srcid,dstid=2,4
                    if arm=='D':
                        srcid=1 if index==0 else 5;dstid=5 if index==0 else 6
                    sptr=48 if fixed else 14
                    regcmd=rcid*0x10000+index*1070*8
                    nxt=regcmd+1070*8 if index+1<count else 0
                    amount=534
                    source=srcid*0x10000+64;dest=dstid*0x10000+64
                    fields=dict(submit_id=seq,index=index,desc_offset=index*40,desc_bytes=40,
                                regcmd=regcmd,amount=1066,words=1070,rc_id=rcid,rc_offset=index*1070*8,
                                fetch_bytes=8560,int_mask=768,sptr_dpu=sptr,sptr_rdma=sptr,
                                src=source,dst=dest,src_id=srcid,dst_id=dstid,data_bytes=data_bytes,
                                next=nxt,next_amount=amount,enable=24,valid=True)
                    emit('program',**fields)
                    def op(target,reg,val):
                        return (target<<48)|(val<<16)|reg
                    words=[0]*1070
                    words[:5]=[op(0x1001,0x4004,sptr),op(0x2001,0x5004,sptr),
                               op(0x2001,0x5018,source),op(0x1001,0x4020,dest),op(0x1001,0x4030,data_bytes//16-1)]
                    words[-4:]=[op(0x101,0x10,nxt) if nxt else 0,op(0x101,0x14,amount),op(0x41,0,0),op(0x81,8,24)]
                    path.joinpath(f'program-{seq:04d}-{index}.bin').write_bytes(struct.pack('<1070Q',*words))
                a=256 if round==0 else 512
                status=256 if fixed else (768-a if arm=='B' and ordinal==0 else a)
                emit('submit_end',submit_id=seq,result=0,errno=0,wall_ns=100000,hw_ns=60000,last_status=status,live=True,descriptor_guard=True)
                destroy(rcid)
        arms=[]
        for arm in 'ABCD':
            submits=2 if arm=='B' else 1
            arms.append(dict(arm=arm,submits=submits,tasks=1 if arm=='A' else 2,
                             irq_delta=[submits,0,0],checks=1 if arm=='A' else 3,max_abs=0,pass_=True))
            arms[-1]['pass']=arms[-1].pop('pass_')
            arms[-1]['padding_guard']=True
        probe.append(dict(event='round',round=round,rows=rows,cols=cols,pattern=round%2,checks=10,arms=arms,pass_=True))
        probe[-1]['pass']=probe[-1].pop('pass_')
        n=rows*cols
        xa=[(i%17)/4-2 for i in range(n)]
        xb=[(i%11)/4-1 for i in range(n)]
        vector(f'round-{round:04d}-input-a.bin',xa);vector(f'round-{round:04d}-input-b.bin',xb)
        ra=[hsig(x) for x in xa];rb=[hsig(x) for x in xb];rd=[hsig(x) for x in ra]
        for arm in 'ABCD':
            vector(f'round-{round:04d}-arm-{arm}-task-0.bin',ra)
            if arm!='A':vector(f'round-{round:04d}-arm-{arm}-task-1.bin',rd if arm=='D' else rb)
    for i in range(1,8):destroy(i)
    emit('end',live_allocations=0)
    probe.append(dict(event='summary',rounds=2,checks=20,failures=0,core_mask=1,sptr=48 if fixed else 14,status='PASS'))
    write_rows(path/'trace.jsonl',events);write_rows(path/'probe.jsonl',probe)
    path.joinpath('probe.log').write_text('CHECK fixture n=512 max_abs=0 bad=0 PASS\n'*20)
    return events,probe


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.trace,self.probe=fixture(self.root)
    def tearDown(self):
        self.temp.cleanup()
    def reject(self,text=''):
        write_rows(self.root/'trace.jsonl',self.trace);write_rows(self.root/'probe.jsonl',self.probe)
        result=verify_process(self.root)
        self.assertEqual(result['status'],'FAIL',result)
        self.assertIn(text,' '.join(result['errors']))
    def test_valid_auto(self):
        self.assertEqual(verify_process(self.root)['status'],'PASS')
    def test_valid_fixed(self):
        fixture(self.root,True)
        self.assertEqual(verify_process(self.root)['status'],'PASS')
    def test_saved_validation_roundtrip_stable(self):
        report=verify_process(self.root)
        self.assertEqual(json.loads(json.dumps(report)),report)
    def test_wrong_link_even_if_raw_matches(self):
        e=next(x for x in self.trace if x['event']=='program' and x['next'])
        e['next']+=16
        p=self.root/f"program-{e['submit_id']:04d}-{e['index']}.bin"
        words=list(struct.unpack('<1070Q',p.read_bytes()))
        words[-4]=(0x101<<48)|(e['next']<<16)|0x10;p.write_bytes(struct.pack('<1070Q',*words))
        self.reject('chain address')
    def test_descriptor_overflow(self):
        e=next(x for x in self.trace if x['event']=='submit_begin' and x['count']==2)
        e['task_start']=819;e['task_offset']=32760
        self.reject('descriptor BO bounds')
    def test_early_destroy(self):
        i=next(i for i,x in enumerate(self.trace) if x['event']=='submit_end')
        self.trace.insert(i,dict(event='destroy',id=1,fd=4,obj=1001,result=0))
        for i,e in enumerate(self.trace,1):e['event_id']=i
        self.reject('active submit')
    def test_bad_irq(self):
        self.probe[0]['arms'][0]['irq_delta']=[2,0,0];self.reject('IRQ completion')
    def test_padding_scribble(self):
        self.probe[0]['arms'][2]['padding_guard']=False;self.reject('cube padding corruption')
    def test_bad_completion(self):
        next(x for x in self.trace if x['event']=='submit_end')['last_status']=16
        self.reject('DPU completion')
    def test_descriptor_scribble(self):
        next(x for x in self.trace if x['event']=='submit_end')['descriptor_guard']=False
        self.reject('descriptor BO changed')
    def test_group_did_not_switch(self):
        next(x for x in self.trace if x['event']=='submit_end' and x['submit_id']==2)['last_status']=256
        self.reject('did not alternate')
    def test_missing_round(self):
        self.probe.pop(1);self.reject('round count')
    def test_missing_return(self):
        self.trace=[x for x in self.trace if not(x['event']=='submit_end' and x['submit_id']==1)]
        for i,e in enumerate(self.trace,1):e['event_id']=i
        self.reject('active submit')
    def test_corrupt_tensor(self):
        p=self.root/'round-0000-arm-D-task-1.bin';data=p.read_bytes()
        p.write_bytes(struct.pack('<e',float('nan'))+data[2:]);self.reject('witness oracle')
    def test_forged_raw_metadata(self):
        next(x for x in self.trace if x['event']=='program')['next_amount']+=1
        self.reject('raw regcmd contradicts')


if __name__=='__main__':
    unittest.main()
