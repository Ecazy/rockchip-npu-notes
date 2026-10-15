#!/usr/bin/env python3
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

source = Path(__file__).resolve().parents[2] / 'tools/python/matched_overlap_analyze.py'
spec = importlib.util.spec_from_file_location('matched', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MatchedEvidence(unittest.TestCase):
    def setUp(self):
        self.case = dict(m=12, k=1024, n=32, b_elements=8192, pairs=1, warmups=4, reps=128)
        self.row = dict(type='matched', arm='overlap', phase='main', rep=0,
                        result=0, valid=1, actual_mask=1, service_ns=100,
                        int_status=256, irq_ok=1, irq=[1, 0, 0],
                        task_count=2, requested_reps=128, **self.case,
                        checks=[dict(tile=0, kind='mm', used=1, valid=1,
                                     elements=384, nan=0, bad=0, hash='0123456789abcdef'),
                                dict(tile=0, kind='lut', used=1, valid=1,
                                     elements=8192, nan=0, bad=0, hash='0123456789abcdef')])

    def test_valid(self):
        self.assertTrue(module.check_sample(self.row, self.case))

    def test_partial_output(self):
        self.row['checks'][1]['nan'] = 8
        self.assertFalse(module.check_sample(self.row, self.case))

    def test_omitted_tile(self):
        case = dict(self.case, pairs=2)
        row = dict(self.row, pairs=2, task_count=4)
        self.assertFalse(module.check_sample(row, case))

    def test_wrong_workload(self):
        row = dict(self.row, k=512)
        self.assertFalse(module.check_sample(row, self.case))

    def test_wrong_submit_or_missing_hash(self):
        row = dict(self.row, task_count=1)
        self.assertFalse(module.check_sample(row, self.case))
        row = copy.deepcopy(self.row)
        row['checks'][0].pop('hash')
        self.assertFalse(module.check_sample(row, self.case))

    def test_inactive_output_must_stay_nan(self):
        row = copy.deepcopy(self.row)
        row.update(arm='b', task_count=1)
        row['checks'][0].update(used=0)
        self.assertFalse(module.check_sample(row, self.case))
        row['checks'][0]['nan'] = 384
        self.assertTrue(module.check_sample(row, self.case))

    def separate_row(self):
        row = copy.deepcopy(self.row)
        row.update(arm='separate', arm_count=9, submits=2, irq=[2, 0, 0],
                   submissions=[dict(tasks=1, service_ns=40, result=0, actual_mask=1, int_status=256),
                                dict(tasks=1, service_ns=60, result=0, actual_mask=1, int_status=512)])
        return row

    def test_two_submits_are_accounted(self):
        self.assertTrue(module.check_sample(self.separate_row(), self.case))

    def test_two_submits_cannot_hide_missing_completion(self):
        row = self.separate_row()
        row['submissions'][0]['result'] = -110
        self.assertFalse(module.check_sample(row, self.case))
        row = self.separate_row()
        row['irq'] = [1, 0, 0]
        self.assertFalse(module.check_sample(row, self.case))

    def test_two_submit_service_sum(self):
        row = self.separate_row()
        row['service_ns'] = 60
        self.assertFalse(module.check_sample(row, self.case))

    def test_streams_only_differ_in_enables(self):
        def enable(mask):
            return (0x81 << 48) | (mask << 16) | 8
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm, masks in [('serial', [0x18, 0x1d]), ('overlap', [0x1d, 0x18])]:
                words = [(0x101 << 48) | (0x10000100 << 16) | 0x10,
                         enable(masks[0]), enable(masks[1])]
                (root / f'case.{arm}.submitted').write_text(
                    '# hw_tasks=2 streams=2 iova=10000000 count=3 pp=0\n' +
                    ''.join(f'{i:04d} {w:016x}\n' for i, w in enumerate(words)))
            self.assertTrue(module.submitted_pair_ok(root, 'case', 1))
            file = root / 'case.overlap.submitted'
            # Change a non-enable command explicitly rather than timing metadata.
            lines = file.read_text().splitlines()
            lines[1] = '0000 0201000000011070'
            file.write_text('\n'.join(lines) + '\n')
            self.assertFalse(module.submitted_pair_ok(root, 'case', 1))


if __name__ == '__main__':
    unittest.main()
