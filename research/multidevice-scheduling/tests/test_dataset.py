import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset import HERE,ROOT,JobCostDataset,confidence,condition,distribution,valid_submit,validate
from collections import Counter

class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.data=JobCostDataset()
    def test_actual_counts_and_domains(self):
        d=self.data.document
        self.assertEqual(validate(d)['submits'],945120)
        self.assertEqual(sum(c['service']['count'] for c in d['cells'] if c['condition']['tracing']),120)
        self.assertEqual(len(d['census']),7)
        self.assertTrue(self.data.match(residency='host_io'))
        self.assertTrue(self.data.match(residency='device_ready'))
        self.assertTrue(self.data.match(workers=8))
        self.assertTrue(self.data.match(task_count=64))
    def test_exact_condition_and_unknown_rejection(self):
        c=self.data.document['cells'][0]['condition']
        self.assertGreater(self.data.cost(c)['duration_ns'],0)
        with self.assertRaises(ValueError):self.data.cost(dict(c,shape_mkn=[999,999,999]))
        with self.assertRaises(ValueError):self.data.cost({'workload':'lut'})
    def test_distribution_and_block_confidence(self):
        self.assertEqual(distribution(Counter({10:3,100:1}))['p50_ns'],10)
        blocks={0:Counter({10:100}),1:Counter({30:100})}
        a=confidence(blocks);self.assertEqual(a,confidence(blocks))
        self.assertEqual(a['mean_block_p50_ns'],20)
        self.assertEqual(a['ci_ns'],[10,30])
        self.assertIsNone(confidence({0:Counter({10:100000})})['ci_ns'])
    def test_raw_filtering_and_corruption(self):
        path=HERE/'evidence/rk3588-trace-20261010/block-00-cell-001/raw.jsonl'
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        r=next(r for r in rows if r['type']=='submit' and not r['warmup'])
        op=next(o for o in rows if o['type']=='operation' and o['operation_id']==r['operation_id'])
        self.assertEqual(valid_submit(r,op),'valid')
        self.assertEqual(valid_submit(dict(r,warmup=True),op),'warmup')
        self.assertEqual(valid_submit(dict(r,guard_ok=False),op),'invalid')
        with self.assertRaises(ValueError):valid_submit(dict(r,driver_service_ns=10**15),op)
    def test_integrity_failure_and_histogram_corruption(self):
        d=copy.deepcopy(self.data.document);d['cells'][0]['service']['count']+=1
        with self.assertRaises(ValueError):validate(d)
        with tempfile.TemporaryDirectory() as folder:
            manifest={'sources':[{'path':'missing','sha256':'0'*64}]}
            with self.assertRaises(ValueError):validate(self.data.document,manifest,folder,True)
