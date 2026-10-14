#!/usr/bin/env python3
"""Evidence failure cases, using a complete independent two-Submit fixture."""
import copy,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'tools/python'))
from device_cost_analyze import program_features,validate_run
from device_cost_campaign import cell
from device_cost_schema import enumerate_config,validate_config
def fixture():
    c=cell('lut',(8,8,8),2,mode='separate');c.update(operations=1,warmup=0,seed=1,timeout_s=30)
    rows=[]
    for i in range(2):
        b=1000+i*5000
        rows.append(dict(type='submit',worker=0,round=0,warmup=False,operation_id=0,request_id=i,submit_index=i,submit_count=2,task_count=1,
            requested_mask=1,actual_mask=1,int_status=512,host_begin_ns=b,ioctl_begin_ns=b+100,ioctl_end_ns=b+3000,host_done_ns=b+3100,
            driver_service_ns=2000,command_bytes=800,estimated_bytes=1536,logical_macs=0,physical_macs=0,result=0,bad_count=0,max_abs_error=.001,guard_ok=True))
    rows += [dict(type='operation',worker=0,round=0,warmup=False,operation_id=0,begin_ns=900,upload_end_ns=900,submit_end_ns=9200,end_ns=9300,
        program_count=2,submit_count=2,completed_submits=2,result=0,bad_count=0,max_abs_error=.001,payload_bytes=3072,allocated_tensor_bo_bytes=70000),
        dict(type='campaign',workers=1,operations=1,warmup_per_worker=0,result=0)]
    return rows,c
class EvidenceTests(unittest.TestCase):
    def reject(self,change):
        r,c=fixture();change(r,c)
        with self.assertRaises((ValueError,IndexError)):validate_run(r,c)
    def test_complete(self):
        r,c=fixture();s,o=validate_run(r,c);self.assertEqual((len(s),len(o)),(2,1))
    def test_missing_submit(self):self.reject(lambda r,c:r.pop(0))
    def test_duplicate_submit(self):self.reject(lambda r,c:r.append(copy.deepcopy(r[0])))
    def test_wrong_core(self):self.reject(lambda r,c:r[0].update(actual_mask=2))
    def test_auto_zero(self):self.reject(lambda r,c:(c.update(strategy='auto'),r[0].update(requested_mask=0,actual_mask=0)))
    def test_service_units(self):self.reject(lambda r,c:r[0].update(driver_service_ns=2000000))
    def test_wrong_features(self):self.reject(lambda r,c:r[0].update(estimated_bytes=0))
    def test_nonfinite(self):self.reject(lambda r,c:r[0].update(max_abs_error=float('inf')))
    def test_numeric_failure(self):self.reject(lambda r,c:r[0].update(bad_count=1))
    def test_program_count(self):self.reject(lambda r,c:r[2].update(program_count=1))
    def test_completion(self):self.reject(lambda r,c:r[0].update(int_status=0))
    def test_boolean_result(self):self.reject(lambda r,c:r[0].update(result=False))
    def test_boolean_error(self):self.reject(lambda r,c:r[0].update(max_abs_error=False))
    def test_nonfinite_operation_error(self):self.reject(lambda r,c:r[2].update(max_abs_error=float('inf')))
    def test_duplicate_operation(self):self.reject(lambda r,c:r.append(copy.deepcopy(r[2])))
    def test_four_programs(self):self.assertEqual(len(program_features(cell('mm_bias_hsig',(256,256,256)))),4)
    def test_separate_budget(self):
        c=cell('mm_bias_hsig',(256,256,256),mode='separate');c.update(operations=12,warmup=2,seed=1,timeout_s=30)
        r=enumerate_config(dict(schema_version=2,blocks=1,cells=[c]));self.assertEqual(r['measured_submits'],48)
    def test_oversize_lut(self):
        with self.assertRaises(ValueError):program_features(cell('lut',(256,256,256)))
    def test_overcommitted_split(self):
        c=cell('lut',(8,8,8),workers=8,strategy='split');c.update(operations=24,warmup=2,seed=1,timeout_s=30)
        validate_config(dict(schema_version=2,blocks=1,cells=[c]))
    def test_deferred_memory_budget(self):
        c=cell('mm_bias_hsig',(256,256,256));c.update(operations=100000,warmup=20,seed=1,timeout_s=30,capture='deferred')
        with self.assertRaises(ValueError):validate_config(dict(schema_version=2,blocks=1,cells=[c]))
if __name__=='__main__':unittest.main()
