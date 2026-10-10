import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from replay import replay_records,replay
from dataset import HERE

def row(rid,start=10,service=5,mask=1):
    return {'type':'submit','request_id':rid,'worker':rid,'round':0,'submit_index':0,
            'warmup':False,'task_count':1,'actual_mask':mask,'host_begin_ns':start-2,
            'ioctl_begin_ns':start,'ioctl_end_ns':start+service+3,'host_done_ns':start+service+5,
            'driver_service_ns':service,'result':0,'bad_count':0,'guard_ok':True}
def bound(r):
    s=r['ioctl_begin_ns'];v=dict(r,driver_entry_ns=[s,s],enqueue_ns=[s,s],start_ns=[s+1,s+1],
                             completion_ns=[s+1+r['driver_service_ns'],s+1+r['driver_service_ns']],queue_ns=[1,1])
    return v
class ReplayTests(unittest.TestCase):
    def test_independent_known_accounting(self):
        a=row(0);r=replay_records([a])['requests'][0]
        self.assertEqual((r['prepare_ns'],r['ioctl_residual_ns'],r['post_ns']),(2,3,2))
        self.assertEqual(r['api_response_ns'],12)
        self.assertIsNone(r['queue_ns'])
    def test_traced_witness_and_multicore(self):
        rows=[row(0,10,5,1),row(1,10,5,2)]
        r=replay_records(rows,list(map(bound,rows)))
        self.assertEqual(r['summary']['synthetic_makespan_ns'],5)
        self.assertTrue(r['summary']['interval_reconstruction_pass'])
        self.assertEqual([x['feasible_start_ns'] for x in r['requests']],[11,11])
    def test_corruption_not_overridden_by_intervals(self):
        a=row(0);b=bound(a)
        with self.assertRaises(ValueError):replay_records([dict(a,host_done_ns=1)],[b])
        with self.assertRaises(ValueError):replay_records([dict(a,driver_service_ns=4)],[b])
        with self.assertRaises(ValueError):replay_records([a,a])
        with self.assertRaises(ValueError):replay_records([a],[b,b])
    def test_overlap_and_impossible_fifo(self):
        a=row(0);b=row(1,11)
        with self.assertRaises(ValueError):replay_records([a,b],[bound(a),bound(b)])
        with self.assertRaises(ValueError):replay_records([dict(a,result=-1)])
    def test_archived_trace_all_120_and_ambiguity(self):
        reports=[replay(p) for p in sorted((HERE/'evidence/rk3588-trace-20261010').glob('block-*'))]
        self.assertEqual(sum(r['summary']['requests'] for r in reports),120)
        self.assertEqual(sum(r['summary']['core_fifo']['overlapping_enqueue_pairs'] for r in reports),3)
        self.assertTrue(all(r['summary']['service_durations_match'] for r in reports))
