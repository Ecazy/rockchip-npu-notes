from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cost_model import CacheIdentity,LinkCost,MemoryBudget,HostStagedMigration,migration_decision,example,measured_cost
from dataset import JobCostDataset
class CostTests(unittest.TestCase):
    def test_link_integer_ceiling_unknown_rejection(self):
        self.assertEqual(LinkCost(3,7).duration(1),333333341)
        for bad in (0,-1,float('nan'),1.1,True):
            with self.assertRaises(Exception):LinkCost(bad,0)
    def test_cache_identity_invalidation(self):
        a=CacheIdentity('A','hash','cube','fp16','1.1.0',0)
        self.assertNotEqual(a,CacheIdentity('B','hash','cube','fp16','1.1.0',0))
        self.assertNotEqual(a,CacheIdentity('A','hash','cube','fp16','1.1.0',1))
    def test_staged_cost_and_both_side_budget(self):
        m=HostStagedMigration(LinkCost(10**9,2),LinkCost(10**9,3),10**9,4,identical_layout=True).estimate(10)
        self.assertEqual(m['duration_ns'],39)
        opts=dict(source_queue_ns=100,source_service_ns=10,target_queue_ns=0,target_service_ns=5,
                  migration=m,source_budget=MemoryBudget(100,staging_bytes=10),target_budget=MemoryBudget(100,staging_bytes=10))
        self.assertTrue(migration_decision(**opts)['migrate'])
        opts['source_budget']=MemoryBudget(5,staging_bytes=10)
        self.assertFalse(migration_decision(**opts)['migrate'])
        with self.assertRaises(ValueError):HostStagedMigration(LinkCost(1,0),LinkCost(1,0),1,0).estimate(1)
    def test_measured_cost_bridge(self):
        data=JobCostDataset();c=data.document['cells'][0]['condition']
        r=measured_cost(c,dataset=data)
        self.assertIsNone(r['cold_start_total_ns'])
        with self.assertRaises(ValueError):measured_cost(c,dataset=data,resident=False)
        doc,results=example(data)
        self.assertEqual(len(results['policies']),4)
        self.assertLess(results['policies']['earliest_finish']['makespan_ns'],results['policies']['round_robin']['makespan_ns'])
