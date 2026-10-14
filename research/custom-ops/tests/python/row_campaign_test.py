#!/usr/bin/env python3
"""The campaign must reject altered, missing, or escaping artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools/python"))
from row_campaign import artifact_report


class ArtifactGateTest(unittest.TestCase):
    def test_integrity_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/"y.bin";data.write_bytes(b"original")
            report={"status":"PASS","artifacts":{"y.bin":hashlib.sha256(data.read_bytes()).hexdigest()}}
            (root/"report.json").write_text(json.dumps(report))
            self.assertEqual(artifact_report(root,"report.json"),report)
            data.write_bytes(b"altered")
            with self.assertRaises(ValueError):artifact_report(root,"report.json")
            data.unlink()
            with self.assertRaises(FileNotFoundError):artifact_report(root,"report.json")

    def test_empty_or_escaping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for artifacts in ({},{"../outside":"bad"},{"/outside":"bad"}):
                (root/"report.json").write_text(json.dumps({"status":"PASS","artifacts":artifacts}))
                with self.assertRaises(ValueError):artifact_report(root,"report.json")


if __name__=="__main__":unittest.main()
