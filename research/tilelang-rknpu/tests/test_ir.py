import json
import unittest
from pathlib import Path

import jsonschema

from rknpu_backend.errors import BackendError, RK_E_SCHEMA, RK_E_SHAPE
from rknpu_backend.ir import Module, matmul_module
from rknpu_backend.serialize import canonical_json, loads

ROOT = Path(__file__).resolve().parents[1]


class IRTests(unittest.TestCase):
    def test_roundtrip_is_canonical(self):
        module = matmul_module(m=13, k=64, n=33, activation="silu")
        dumped = canonical_json(module.to_dict())
        again = Module.from_dict(json.loads(dumped))
        self.assertEqual(canonical_json(again.to_dict()), dumped)

    def test_duplicate_json_key_rejected(self):
        with self.assertRaises(BackendError) as ctx:
            loads('{"a":1,"a":2}')
        self.assertEqual(ctx.exception.code, RK_E_SCHEMA)

    def test_nan_and_infinity_rejected(self):
        for text in ('{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'):
            with self.assertRaises(BackendError):
                loads(text)

    def test_unknown_field_rejected(self):
        obj = matmul_module(m=2, k=3, n=4).to_dict()
        obj["surprise"] = 1
        with self.assertRaises(BackendError):
            Module.from_dict(obj)

    def test_bool_is_not_an_integer_shape(self):
        obj = matmul_module(m=2, k=3, n=4).to_dict()
        obj["values"][0]["type"]["shape"][0] = True
        with self.assertRaises(BackendError):
            Module.from_dict(obj)

    def test_non_contiguous_value_ids_rejected(self):
        obj = matmul_module(m=2, k=3, n=4).to_dict()
        obj["values"][1]["id"] = 7
        with self.assertRaises(BackendError):
            Module.from_dict(obj)

    def test_rkir_schema_accepts_real_module(self):
        schema = json.loads((ROOT / "schema" / "rkir.v1.json").read_text())
        jsonschema.validate(matmul_module(m=2, k=3, n=4).to_dict(), schema)

    def test_nonzero_matmul_initial_value_is_not_representable(self):
        # The MVP IR has no initial-C operand.  The frontend test covers the real
        # nonzero-fill negative case; here we assert matmul arity cannot hide one.
        obj = matmul_module(m=2, k=3, n=4).to_dict()
        obj["ops"][0]["inputs"].append(0)
        with self.assertRaises(BackendError):
            Module.from_dict(obj)


if __name__ == "__main__":
    unittest.main()
