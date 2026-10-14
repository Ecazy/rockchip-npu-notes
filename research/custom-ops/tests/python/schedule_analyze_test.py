#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
sys.path.insert(0,str(_Path(__file__).resolve().parents[2]/"tools/python"))

import copy,json,pathlib,unittest
from schedule_analyze import analyze
ROOT=pathlib.Path(__file__).resolve().parents[2]/'tests/scheduling'
class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.rows=[json.loads(l) for l in (ROOT/'serial.jsonl').read_text().splitlines()]
        self.trace=(ROOT/'serial.trace').read_text();self.stats=json.loads((ROOT/'stats.json').read_text())
    def rejects(self,rows=None,trace=None,stats=None):
        with self.assertRaises(ValueError):analyze(rows or self.rows,self.trace if trace is None else trace,self.stats if stats is None else stats)
    def test_single_job(self):
        p=analyze(self.rows,self.trace,self.stats)[0]
        self.assertEqual(p['irq_core'],0);self.assertGreater(p['queue_ns'][0],0)
        self.assertLess(p['queue_ns'][1]-p['queue_ns'][0],10000)
        self.assertEqual(p['response_ns'],p['ioctl_end_ns']-p['host_begin_ns'])
        for q,other in zip(p['queue_ns'],reversed(p['other_ns'])):
            self.assertEqual(q+other+p['driver_service_ns'],p['response_ns'])
    def test_fifo_irq_handoff(self):
        rows=[json.loads(l) for l in (ROOT/'same.jsonl').read_text().splitlines()]
        out=analyze(rows,(ROOT/'same.trace').read_text(),self.stats)
        self.assertEqual(len(out),4);self.assertTrue(any(p['queue_ns'][0]>500000 for p in out))
        self.assertEqual({p['irq_core'] for p in out},{0})
    def test_same_context_timestamp_tie(self):
        trace=self.trace.replace('32728.197846','32728.197856')
        analyze(self.rows,trace,self.stats)
    def test_cross_cpu_timestamp_tie(self):
        line=next(l for l in self.trace.splitlines() if '32728.197846' in l)
        new=line.replace('32728.197846','32728.197856').replace('1)','2)')
        self.rejects(trace=self.trace.replace(line,new))
    def test_missing_cpu_stats(self):self.rejects(stats={})
    def test_trace_loss(self):self.rejects(stats={'cpu0':'overrun: 1\ncommit overrun: 0\ndropped events: 0\n'})
    def test_missing_stats(self):self.rejects(stats={'cpu0':'overrun: 0\n'})
    def test_missing_marker(self):self.rejects(trace='\n'.join(l for l in self.trace.splitlines() if 'RS done' not in l))
    def test_wrong_irq_core(self):self.rejects(trace=self.trace.replace('rknpu_core0_irq_handler','rknpu_core1_irq_handler'))
    def test_missing_completion_stamp(self):
        self.rejects(trace='\n'.join(l for l in self.trace.splitlines() if not ('32728.197936' in l)))
    def test_service_clock_or_pair_mismatch(self):
        rows=copy.deepcopy(self.rows);rows[0]['driver_service_ns']+=100000;self.rejects(rows=rows)
    def test_wrong_binding(self):
        rows=copy.deepcopy(self.rows);rows[0]['actual_mask']=2;self.rejects(rows=rows)
    def test_ganged_mask(self):
        rows=copy.deepcopy(self.rows);rows[0]['actual_mask']=3;self.rejects(rows=rows)
    def test_candidate_set(self):
        rows=copy.deepcopy(self.rows);rows[0]['allowed_mask']=2;self.rejects(rows=rows)
    def test_duplicate_job(self):self.rejects(rows=self.rows+self.rows)
    def test_numeric_failure(self):
        rows=copy.deepcopy(self.rows);rows[0]['bad']=1;self.rejects(rows=rows)
    def test_duplicate_commit_stamp(self):
        line=next(l for l in self.trace.splitlines() if '32728.197856' in l)
        self.rejects(trace=self.trace.replace(line,line+'\n'+line))
    def test_distinct_clock_candidates_enclosed(self):
        line=next(l for l in self.trace.splitlines() if '32728.197856' in l)
        candidate=line.replace('32728.197856','32728.197857')
        out=analyze(self.rows,self.trace.replace(line,line+'\n'+candidate),self.stats)[0]
        reference=analyze(self.rows,self.trace,self.stats)[0]
        self.assertLessEqual(out['start_ns'][0],reference['start_ns'][0])
        self.assertGreaterEqual(out['start_ns'][1],reference['start_ns'][1])
if __name__=='__main__':unittest.main()
