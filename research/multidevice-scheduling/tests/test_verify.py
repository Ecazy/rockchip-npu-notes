from pathlib import Path
import sys
import tempfile
import shutil
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from verify import verify
from dataset import ROOT
class VerifyTests(unittest.TestCase):
    def test_portable_verification(self):
        r=verify();self.assertEqual(r['dataset']['submits'],945120)
        self.assertFalse(r['dataset']['full_sources_verified'])
        self.assertEqual(sum(x['requests'] for x in r['archived_trace_replays']),120)
    def test_changed_snapshot_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            dst=Path(folder)/'research/multidevice-scheduling'
            shutil.copytree(ROOT/'research/multidevice-scheduling/static-analysis',dst/'static-analysis')
            p=dst/'static-analysis/sources/rknn3-toolkit/README_CN.md';p.write_text('corrupt')
            with self.assertRaises(ValueError):verify(folder)
