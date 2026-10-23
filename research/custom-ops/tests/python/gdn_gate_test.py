#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('gate', Path(__file__).resolve().parents[2] / 'tools/python/gdn_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)

class GateTests(unittest.TestCase):
    def rows(self):
        rows = []
        for dv, dk in gate.SHAPES:
            base = dict(dv=dv, dk=dk)
            for kind, arms in (('gemv', ('A', 'A2', 'B', 'C', 'A_pair')), ('outer', ('K1', 'EW_expanded'))):
                for arm in arms:
                    n = 0 if arm == 'C' else 2 if arm == 'A_pair' else 1
                    rows.append(dict(base, kind=kind, arm=arm, rep=0, total_us=1., tasks=n, submits=n,
                                     cpu_kind='aarch64_neon_f16_storage_f32_acc'))
            rows.append(dict(base, kind='resident_repeat', steps=gate.STEPS, state_exports_during_steps=0, bit_exact=True))
            rows.append(dict(base, kind='scalar_and_subtract', scales=4))
            rows.extend(dict(base, kind='recurrent', token=t, tasks=6, submits=1) for t in range(gate.STEPS))
        rows.append(dict(kind='result', status='PASS'))
        return rows

    def run_rows(self, rows):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'run.log'
            p.write_text('\n'.join(map(json.dumps, rows)))
            return gate.logs(p, 1, 'chain')

    def test_valid(self):
        self.assertEqual(len(self.run_rows(self.rows())), len(gate.SHAPES)*7)

    def test_corruption(self):
        for kind in ('missing', 'duplicate', 'nan', 'cpu', 'tasks', 'resident', 'no_pass'):
            with self.subTest(kind=kind):
                rows = self.rows()
                if kind == 'missing': rows.pop(0)
                if kind == 'duplicate': rows.append(rows[0])
                if kind == 'nan': rows[0]['total_us'] = float('nan')
                if kind == 'cpu': rows[0]['cpu_kind'] = 'portable'
                if kind == 'tasks': rows[0]['tasks'] = 0
                if kind == 'resident': rows[7]['state_exports_during_steps'] = 1
                if kind == 'no_pass': rows.pop()
                with self.assertRaises(ValueError): self.run_rows(rows)

    def test_numerical_rejection(self):
        for value in (float('nan'), float('inf'), 1.):
            with self.assertRaises(ValueError): gate.check(value, 0.)

if __name__ == '__main__':
    unittest.main()
