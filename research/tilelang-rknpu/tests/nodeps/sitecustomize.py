"""Subprocess helper: make the optional TileLang/TVM tree unimportable.

The refactor spec requires that base package import, legacy RKIR compilation and
plan replay keep working without TileLang/TVM.  Tests add this directory to
``PYTHONPATH`` in a child interpreter so the absence is enforced rather than
assumed.
"""
import sys

_BLOCKED = ("tilelang", "tvm", "tvm_ffi", "ml_dtypes")


class _Blocker:
    def find_module(self, name, path=None):  # pragma: no cover - legacy API
        return self if self._blocked(name) else None

    def find_spec(self, name, path=None, target=None):
        if self._blocked(name):
            raise ImportError(f"{name} is deliberately unavailable in this test")
        return None

    @staticmethod
    def _blocked(name):
        return any(name == b or name.startswith(b + ".") for b in _BLOCKED)


for _name in [m for m in sys.modules if m.split(".")[0] in _BLOCKED]:
    del sys.modules[_name]
sys.meta_path.insert(0, _Blocker())
