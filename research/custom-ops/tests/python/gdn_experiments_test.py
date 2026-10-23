#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('gate',Path(__file__).resolve().parents[2]/'tools/python/gdn_experiments_gate.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
class Evidence(unittest.TestCase):
    def test_fp32_is_stricter(self):
        g.error([1.0002],[1.0])
        with self.assertRaises(ValueError):g.error([1.0002],[1.0],True)
        with self.assertRaises(ValueError):g.error([float('nan')],[0.])
    def test_missing_or_fake_run(self):
        with self.assertRaises(ValueError):g.validate_rows([])
        with self.assertRaises(ValueError):g.validate_rows([dict(kind='result',status='PASS'),dict(kind='backend',cpu='portable-f32-fma')])
        with self.assertRaises(ValueError):g.validate_rows([dict(kind='result',status='PASS'),dict(kind='backend',cpu='aarch64-neon-f32-fma')])
    def test_graph_task_contract(self):
        self.assertEqual(g.expected_tasks('merged',128),7)
        self.assertEqual(g.expected_tasks('beta_fold',128),5)
        self.assertEqual(g.expected_tasks('native_alpha',256),7)
    def test_clock_fields_are_required(self):
        base=dict(kind='step',n=128,block=0,token=0,arm='npu_alloc',instrumented=0,
                  tasks=6,submits=1,total_us=1.0,freq_before_khz=2304000,
                  freq_after_khz=2304000,npu_freq_before_khz=1000000,
                  npu_freq_after_khz=1000000)
        rows=[dict(kind='result',status='PASS'),dict(kind='backend',cpu='aarch64-neon-f32-fma'),base]
        # A single step is not a complete campaign, but the clock-field check
        # must be reached before completeness can fail for a different reason.
        with self.assertRaises(ValueError) as ctx:
            g.validate_rows(rows)
        self.assertNotIn('missing NPU frequency',str(ctx.exception))
        base.pop('npu_freq_before_khz')
        with self.assertRaisesRegex(ValueError,'missing NPU frequency'):
            g.validate_rows(rows)
if __name__=='__main__':unittest.main()
