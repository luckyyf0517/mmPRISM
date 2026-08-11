import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmprism.data import (
    CslNewsAnnotationConfig,
    build_csl_news_annotation_identity_audit,
    load_csl_news_annotation_config,
    write_csl_news_annotation_identity_audit,
)
from mmprism.data import csl_news_annotation_audit as audit_module


class CslNewsAnnotationIdentityAuditTest(unittest.TestCase):
    def _runtime(self, *, dirty: bool = False) -> dict[str, object]:
        return {"git": {"commit": "f" * 40, "dirty": dirty}}

    def _config(self, root: Path) -> CslNewsAnnotationConfig:
        path = root / "annotation.yaml"
        path.write_text(
            f"""schema_version: mmprism.csl_news_pose_annotation.v1
source:
  archive_root: {root / 'archives'}
  labels_path: {root / 'labels.json'}
  source_id: fixture
  source_revision: revision
  expected_archive_count: 1
model:
  mmpose_root: {root / 'mmpose'}
  mmpose_commit: commit
  project_dir: {root / 'project'}
  config_path: {root / 'model.py'}
  checkpoint_path: {root / 'model.pth'}
  checkpoint_sha256: {'a' * 64}
  device: cuda:0
transform:
  crop_top: 20
  crop_left: 20
  crop_right: 20
  confidence_threshold: 0.3
runtime:
  output_root: {root / 'output'}
  scratch_root: {root / 'scratch'}
  worker_index: 0
  worker_count: 1
  poll_seconds: 60
  min_free_bytes: 0
  max_consecutive_oom: 2
""",
            encoding="utf-8",
        )
        return load_csl_news_annotation_config(path, root)

    def _write_pair(
        self,
        config: CslNewsAnnotationConfig,
        sample_id: str,
        *,
        artifact_bytes: bytes = b"pose-artifact",
        declared_size: int | None = None,
        declared_sha256: str | None = None,
    ) -> tuple[Path, Path]:
        sample_root = config.runtime.output_root / "samples" / "archive_001"
        sample_root.mkdir(parents=True, exist_ok=True)
        artifact = sample_root / f"{sample_id}.npz"
        artifact.write_bytes(artifact_bytes)
        sidecar = artifact.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": "mmprism.csl_news_pose_sample.v1",
                    "status": "completed",
                    "sample_id": sample_id,
                    "config_fingerprint": config.fingerprint,
                    "source": {"archive": "archive_001.zip"},
                    "artifact": {
                        "path": str(artifact),
                        "size_bytes": (
                            len(artifact_bytes)
                            if declared_size is None
                            else declared_size
                        ),
                        "sha256": (
                            hashlib.sha256(artifact_bytes).hexdigest()
                            if declared_sha256 is None
                            else declared_sha256
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )
        return artifact, sidecar

    def test_stream_audits_all_frozen_pairs_without_reading_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            self._write_pair(config, "sample-a", artifact_bytes=b"a")
            self._write_pair(config, "sample-b", artifact_bytes=b"bb")
            hidden = (
                config.runtime.output_root
                / "samples"
                / "archive_001"
                / ".sample-c.json"
            )
            hidden.write_text("{}", encoding="utf-8")
            report = build_csl_news_annotation_identity_audit(
                config, runtime_report=self._runtime()
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["scope"]["frozen_sidecar_count"], 2)
        self.assertEqual(report["summary"]["passed_pair_count"], 2)
        self.assertEqual(report["summary"]["artifact_bytes_hashed"], 3)
        self.assertEqual(report["invalid_pairs"], [])

    def test_reports_every_invalid_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            first_artifact, _ = self._write_pair(
                config,
                "sample-a",
                declared_size=0,
                declared_sha256=hashlib.sha256(b"").hexdigest(),
            )
            second_artifact, _ = self._write_pair(config, "sample-b")
            second_artifact.unlink()
            report = build_csl_news_annotation_identity_audit(
                config, runtime_report=self._runtime()
            )
            preserved_bytes = first_artifact.read_bytes()

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["summary"]["failed_pair_count"], 2)
        failures_by_sample = {
            item["sample_id"]: item["failures"] for item in report["invalid_pairs"]
        }
        self.assertIn("artifact_size_mismatch", failures_by_sample["sample-a"])
        self.assertIn("artifact_sha256_mismatch", failures_by_sample["sample-a"])
        self.assertIn("artifact_unreadable_or_missing", failures_by_sample["sample-b"])
        self.assertEqual(preserved_bytes, b"pose-artifact")

    def test_freezes_sidecar_list_before_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            self._write_pair(config, "sample-a")
            original = audit_module._audit_annotation_pair

            def add_late_pair(*args: object, **kwargs: object):
                self._write_pair(config, "sample-late")
                return original(*args, **kwargs)

            with patch.object(
                audit_module, "_audit_annotation_pair", side_effect=add_late_pair
            ):
                report = build_csl_news_annotation_identity_audit(
                    config, runtime_report=self._runtime()
                )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["scope"]["frozen_sidecar_count"], 1)
        self.assertEqual(report["summary"]["audited_pair_count"], 1)

    def test_detects_artifact_stat_change_during_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            self._write_pair(config, "sample-a")
            original = audit_module._stream_sha256

            def mutate_after_hash(path: Path) -> tuple[str, int]:
                result = original(path)
                path.write_bytes(path.read_bytes() + b"changed")
                return result

            with patch.object(
                audit_module, "_stream_sha256", side_effect=mutate_after_hash
            ):
                report = build_csl_news_annotation_identity_audit(
                    config, runtime_report=self._runtime()
                )

        self.assertIn(
            "artifact_changed_during_audit",
            report["invalid_pairs"][0]["failures"],
        )

    def test_empty_scope_fails_and_report_write_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            report = build_csl_news_annotation_identity_audit(
                config, runtime_report=self._runtime()
            )
            output = root / "audits" / "audit.json"
            written = write_csl_news_annotation_identity_audit(report, output)
            payload = json.loads(written.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["audit_failures"], ["no_visible_sidecars"])
        self.assertEqual(payload["schema_version"], report["schema_version"])


if __name__ == "__main__":
    unittest.main()
