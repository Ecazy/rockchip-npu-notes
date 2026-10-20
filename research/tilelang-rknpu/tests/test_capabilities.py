import json
import unittest
from pathlib import Path

import jsonschema

from rknpu_backend.capabilities import default_profile
from rknpu_backend.errors import BackendError, RK_E_UNSUPPORTED


class CapabilityTests(unittest.TestCase):
    def test_default_profile_hash_is_stable(self):
        a = default_profile()
        b = default_profile()
        self.assertEqual(a.profile_hash, b.profile_hash)
        self.assertEqual(a.chip_id, "rk3588")
        self.assertEqual(a.max_programs_per_submit, 64)

    def test_experimental_capability_is_not_default(self):
        profile = default_profile()
        with self.assertRaises(BackendError) as ctx:
            profile.enabled("co.scale_native.f16.v1")
        self.assertEqual(ctx.exception.code, RK_E_UNSUPPORTED)

    def test_capability_schema(self):
        root = Path(__file__).resolve().parents[1]
        schema = json.loads((root / "schema" / "capability.v1.json").read_text())
        manifest = json.loads((root / "capabilities" /
                               "rk3588_vendor_co_v1.json").read_text())
        jsonschema.validate(manifest, schema)

    def test_enabled_capability(self):
        profile = default_profile()
        cap = profile.enabled("co.mm.f16.v1")
        self.assertEqual(cap.status, "enabled")
        self.assertEqual(cap.target_profile, "rk3588_vendor_co_v1")


if __name__ == "__main__":
    unittest.main()
