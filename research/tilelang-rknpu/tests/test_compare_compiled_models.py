"""The model-package comparator must fail on an incomplete or edited package.

The comparator is evidence-producing code: a tool that only walks the candidate and
trusts recorded digests can report "identity only" for a package with deleted files
or an edited manifest.  Every case here builds a tiny synthetic package pair and
asserts the tool's exit status and classification, so the mechanism is tested
independently of the real ``models/compiled`` content.
"""
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_tool():
    path = ROOT / "tools" / "compare_compiled_models.py"
    spec = importlib.util.spec_from_file_location("_cmp_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARATOR = _load_tool()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PackageBuilder:
    """A minimal but faithfully shaped compiled model package."""

    def __init__(self, root: Path):
        # ``root`` is the package's ``compiled`` directory parent, i.e. the real
        # ``models/`` directory; ``source_files`` paths are repo-root relative, so
        # they resolve through ``compiled_dir.parents[1]`` exactly as in production.
        self.root = root
        (root / "compiled").mkdir(parents=True, exist_ok=True)
        # Mirrors the production layout: <repo>/models/compiled, with
        # ``source_files`` keys relative to <repo>.
        self.source_root = root.parents[0]
        (self.source_root / "models").mkdir(parents=True, exist_ok=True)
        (self.source_root / "models" / "kernel.py").write_text("source\n")

    @property
    def compiled(self) -> Path:
        return self.root / "compiled"

    def plan(self, name: str, source_hash: str = "Span(SourceName(a.py, 0x1), 1)",
             tasks: int = 1) -> Path:
        plan = {
            "kind": "rkplan", "name": name, "source_hash": source_hash,
            "plan_hash": hashlib.sha256(f"{name}{source_hash}{tasks}".encode()).hexdigest(),
            "actions": [{"kind": "npu_task", "id": i} for i in range(tasks)],
        }
        path = self.compiled / f"{name}.rkplan.json"
        path.write_text(json.dumps(plan, sort_keys=True))
        return path

    def hybrid(self, name: str, plans: dict) -> Path:
        payload = {"kind": "rkhybrid", "plans": plans}
        path = self.compiled / f"{name}.rkhybrid.json"
        path.write_text(json.dumps(payload, sort_keys=True))
        return path

    def manifest(self, plans: dict, hybrids: dict, diagnostics: dict | None = None):
        path = self.compiled / "manifest.json"
        payload = {
            "schema_version": 1,
            "plans": plans,
            "hybrid_manifests": hybrids,
            "mlp_hidden_diagnostics": diagnostics or {},
            "source_files": {
                "models/kernel.py": _sha(self.source_root / "models" / "kernel.py")},
        }
        path.write_text(json.dumps(payload, sort_keys=True))
        return path


def build_package(root: Path, *, source_hash: str = "Span(SourceName(a.py, 0x1), 1)"):
    builder = PackageBuilder(root)
    plan_path = builder.plan("main", source_hash=source_hash)
    plan = json.loads(plan_path.read_text())
    diag_path = builder.plan("diag", source_hash=source_hash, tasks=3)
    diag = json.loads(diag_path.read_text())
    hybrid_path = builder.hybrid(
        "model", {"main": {"path": plan_path.name, "plan_hash": plan["plan_hash"],
                           "source_hash": plan["source_hash"]}})
    builder.manifest(
        plans={
            "main": {"path": plan_path.name, "plan_hash": plan["plan_hash"],
                     "source_hash": plan["source_hash"]},
            "diag": {"path": diag_path.name, "plan_hash": diag["plan_hash"],
                     "source_hash": diag["source_hash"]},
        },
        hybrids={"model": {"path": hybrid_path.name, "sha256": _sha(hybrid_path)}},
        diagnostics={"diag": {"plan_hash": diag["plan_hash"], "role": "r"}},
    )
    return builder


class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        base = Path(self._tmp.name)
        self.reference = build_package(base / "ref-pkg" / "models").compiled
        self.candidate = build_package(base / "cand-pkg" / "models").compiled
        self.out = base / "comparison.json"

    def run_tool(self):
        status = COMPARATOR.main(["--reference", str(self.reference),
                                  "--candidate", str(self.candidate),
                                  "--out", str(self.out)])
        return status, json.loads(self.out.read_text())

    def test_identical_packages_pass(self):
        status, report = self.run_tool()
        self.assertEqual(status, 0)
        self.assertEqual(report["failures"], [])
        self.assertEqual(set(report["counts"]), {"identical"})

    def test_volatile_source_hash_is_an_identity_difference(self):
        other = build_package(Path(self._tmp.name) / "vol-pkg" / "models",
                              source_hash="Span(SourceName(a.py, 0xdead), 1)")
        self.candidate = other.compiled
        status, report = self.run_tool()
        self.assertEqual(status, 0, report["failures"])
        # The plan hashes embed the address, so the manifest cascades.
        self.assertIn("identity-cascade", report["counts"])
        self.assertEqual(report["reference_problems"], {})

    def test_deleted_referenced_hybrid_fails(self):
        (self.candidate / "model.rkhybrid.json").unlink()
        status, report = self.run_tool()
        self.assertNotEqual(status, 0)
        self.assertTrue(report["failures"])
        self.assertIn("missing-candidate", json.dumps(report["files"]))
        self.assertTrue(any("hybrid_manifests.model.path missing"
                            in problem
                            for problems in report["reference_problems"].values()
                            for problem in problems))

    def test_deleted_unreferenced_plan_fails(self):
        (self.candidate / "diag.rkplan.json").unlink()
        status, report = self.run_tool()
        self.assertNotEqual(status, 0)
        self.assertEqual(report["files"]["diag.rkplan.json"]["verdict"],
                         "missing-candidate")

    def test_extra_candidate_file_fails(self):
        extra = self.candidate / "extra.rkplan.json"
        extra.write_text("{}")
        status, report = self.run_tool()
        self.assertNotEqual(status, 0)
        self.assertEqual(report["files"]["extra.rkplan.json"]["verdict"],
                         "missing-reference")

    def test_edited_hybrid_digest_fails(self):
        manifest_path = self.candidate / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["hybrid_manifests"]["model"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        status, report = self.run_tool()
        self.assertNotEqual(status, 0)
        self.assertTrue(any("hybrid_manifests.model.sha256 does not match"
                            in problem
                            for problems in report["reference_problems"].values()
                            for problem in problems))

    def test_tampered_plan_file_fails(self):
        plan_path = self.candidate / "main.rkplan.json"
        plan = json.loads(plan_path.read_text())
        plan["actions"].append({"kind": "npu_task", "id": 99})
        plan_path.write_text(json.dumps(plan, sort_keys=True))
        status, report = self.run_tool()
        self.assertNotEqual(status, 0)
        self.assertEqual(report["files"]["main.rkplan.json"]["verdict"],
                         "content-diff")
        self.assertTrue(report["failures"])

    def test_plan_file_edited_and_manifest_digest_rewritten_fails(self):
        """Editing a plan *and* its recorded hash must not slip through.

        The referenced file is then content-identical to nothing, so the manifest's
        digest no longer describes the plan it points at.
        """
        plan_path = self.candidate / "main.rkplan.json"
        plan = json.loads(plan_path.read_text())
        plan["actions"] = []
        plan["plan_hash"] = "b" * 64
        plan_path.write_text(json.dumps(plan, sort_keys=True))
        status, report = self.run_tool()
        self.assertNotEqual(status, 0)
        self.assertTrue(any("plans.main.plan_hash does not match its file"
                            in problem
                            for problems in report["reference_problems"].values()
                            for problem in problems))

    def test_manifest_plan_hash_must_match_the_plan_file(self):
        plan_path = self.candidate / "main.rkplan.json"
        plan = json.loads(plan_path.read_text())
        plan_path.write_text(json.dumps(plan, sort_keys=True))  # unchanged
        manifest_path = self.candidate / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["plans"]["main"]["plan_hash"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        status, report = self.run_tool()
        self.assertNotEqual(status, 0)
        self.assertTrue(any("plans.main.plan_hash does not match"
                            in problem
                            for problems in report["reference_problems"].values()
                            for problem in problems))

    def test_edited_source_digest_fails(self):
        manifest_path = self.candidate / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_files"]["models/kernel.py"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest, sort_keys=True))
        status, report = self.run_tool()
        self.assertNotEqual(status, 0)
        self.assertTrue(any("source_files.models/kernel.py digest does not match"
                            in problem
                            for problems in report["reference_problems"].values()
                            for problem in problems))

    def test_missing_reference_directory_is_an_error(self):
        with self.assertRaises(SystemExit):
            COMPARATOR.main(["--reference", str(self.candidate / "nope"),
                             "--candidate", str(self.candidate),
                             "--out", str(self.out)])


if __name__ == "__main__":
    unittest.main()
