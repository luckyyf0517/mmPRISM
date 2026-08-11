import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from mmprism.data import (
    CslNewsAnnotationConfig,
    build_csl_news_annotation_status,
    load_csl_news_annotation_config,
    write_csl_news_annotation_status,
)


class CslNewsAnnotationStatusTest(unittest.TestCase):
    def _config(self, root: Path) -> CslNewsAnnotationConfig:
        path = root / "status.yaml"
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

    def _write_fixture(self, root: Path) -> tuple[CslNewsAnnotationConfig, Path]:
        config = self._config(root)
        config.source.archive_root.mkdir(parents=True)
        with zipfile.ZipFile(config.source.archive_root / "archive_001.zip", "w") as archive:
            archive.writestr("first.mp4", b"video-one")
            archive.writestr("second.mp4", b"video-two")

        sample_root = config.runtime.output_root / "samples" / "archive_001"
        sample_root.mkdir(parents=True)
        artifact_path = sample_root / "sample.npz"
        frame_count = 2
        np.savez_compressed(
            artifact_path,
            native_keypoints_3d=np.zeros((frame_count, 133, 3), dtype=np.float32),
            native_keypoint_scores=np.ones((frame_count, 133), dtype=np.float32),
            transformed_keypoints_2d=np.zeros((frame_count, 133, 2), dtype=np.float32),
            frame_indices=np.arange(frame_count, dtype=np.int64),
            timestamps_seconds=np.arange(frame_count, dtype=np.float64) / 25,
            canonical_pose=np.zeros((frame_count, 2, 24, 3), dtype=np.float32),
            canonical_confidence=np.ones((frame_count, 2, 24), dtype=np.float32),
            canonical_valid=np.ones((frame_count, 2, 24), dtype=np.bool_),
        )
        checksum = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        (sample_root / "sample.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "generated_at": "2026-08-11T14:00:00+00:00",
                    "sample_id": "sample",
                    "config_fingerprint": config.fingerprint,
                    "annotation": {"text": "测试文本"},
                    "video": {"decoded_frame_count": frame_count},
                    "elapsed_seconds": 1.0,
                    "artifact": {"sha256": checksum},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        run_root = config.runtime.output_root / "runs"
        run_root.mkdir(parents=True)
        (run_root / "run_fixture.json").write_text(
            json.dumps({"started_at": "2026-08-11T13:59:00+00:00"}),
            encoding="utf-8",
        )
        return config, artifact_path

    def test_reports_progress_and_validates_recent_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, artifact_path = self._write_fixture(Path(directory))
            artifact_path.with_name(".sample.npz.tmp.123.npz").write_bytes(b"partial")
            report = build_csl_news_annotation_status(config, sample_validate_count=1)

        self.assertEqual(report["status"], "healthy")
        self.assertEqual(report["source"]["complete_archive_count"], 1)
        self.assertEqual(report["source"]["available_video_count"], 2)
        self.assertEqual(report["annotation"]["completed_sample_count"], 1)
        self.assertEqual(report["annotation"]["npz_count"], 1)
        self.assertEqual(report["annotation"]["remaining_available_sample_count"], 1)
        self.assertEqual(report["sample_validation"]["passed"], 1)

    def test_selects_one_valid_recovery_variant_over_invalid_canonical_pair(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, canonical_artifact = self._write_fixture(root)
            archive_path = config.source.archive_root / "archive_001.zip"
            archive_stat = archive_path.stat()
            archive_sha256 = "a" * 64
            labels_sha256 = "c" * 64
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "mmprism.csl_news_source_integrity_registry.v1"
                        ),
                        "source": {
                            "source_id": "fixture",
                            "source_revision": "revision",
                            "labels_sha256": labels_sha256,
                        },
                        "summary": {
                            "passed_count": 1,
                            "failed_count": 0,
                            "pending_count": 0,
                            "missing_count": 0,
                        },
                        "archives": {
                            "001": {
                                "archive_id": "001",
                                "archive_name": "archive_001.zip",
                                "status": "passed",
                                "source_present": True,
                                "size_bytes": archive_stat.st_size,
                                "mtime_ns": archive_stat.st_mtime_ns,
                                "sha256": archive_sha256,
                                "video_count": 2,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            canonical_sidecar = canonical_artifact.with_suffix(".json")
            canonical_payload = json.loads(
                canonical_sidecar.read_text(encoding="utf-8")
            )
            canonical_payload["source"] = {
                "integrity": {
                    "archive_sha256": archive_sha256,
                    "labels_sha256": labels_sha256,
                }
            }
            canonical_payload["artifact"]["sha256"] = "0" * 64
            canonical_sidecar.write_text(
                json.dumps(canonical_payload), encoding="utf-8"
            )

            variant_stem = f"sample--source_{archive_sha256}"
            variant_artifact = canonical_artifact.with_name(f"{variant_stem}.npz")
            variant_artifact.write_bytes(canonical_artifact.read_bytes())
            variant_sidecar = canonical_sidecar.with_name(f"{variant_stem}.json")
            variant_payload = dict(canonical_payload)
            variant_payload["artifact"] = {
                "sha256": hashlib.sha256(variant_artifact.read_bytes()).hexdigest()
            }
            variant_sidecar.write_text(
                json.dumps(variant_payload), encoding="utf-8"
            )

            report = build_csl_news_annotation_status(
                config,
                sample_validate_count=1,
                integrity_registry_path=registry,
            )

        self.assertEqual(report["status"], "healthy")
        self.assertEqual(report["annotation"]["completed_sample_count"], 1)
        self.assertEqual(
            report["annotation"]["recovered_current_source_sample_count"], 1
        )
        self.assertEqual(
            report["annotation"]["shadowed_invalid_current_source_sidecar_count"],
            1,
        )
        self.assertEqual(
            report["annotation"]["duplicate_current_source_sample_count"], 0
        )
        self.assertEqual(
            report["sample_validation"]["samples"][0]["sidecar"],
            str(variant_sidecar),
        )

    def test_reports_missing_sidecar_as_attention_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, artifact_path = self._write_fixture(Path(directory))
            artifact_path.with_suffix(".json").unlink()
            report = build_csl_news_annotation_status(config, sample_validate_count=0)

        self.assertEqual(report["status"], "attention_required")
        self.assertEqual(report["annotation"]["missing_sidecar_count"], 1)

    def test_writes_status_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports" / "status.json"
            written = write_csl_news_annotation_status({"status": "healthy"}, output)
            payload = json.loads(written.read_text(encoding="utf-8"))

        self.assertEqual(payload, {"status": "healthy"})

    def test_integrity_registry_limits_available_archives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self._write_fixture(root)
            second_archive = config.source.archive_root / "archive_002.zip"
            with zipfile.ZipFile(second_archive, "w") as archive:
                archive.writestr("excluded.mp4", b"not-eligible")
            excluded_sample_root = config.runtime.output_root / "samples" / "archive_002"
            excluded_sample_root.mkdir(parents=True)
            (excluded_sample_root / "excluded.npz").write_bytes(b"quarantined")
            (excluded_sample_root / "excluded.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "config_fingerprint": config.fingerprint,
                    }
                ),
                encoding="utf-8",
            )
            first_archive = config.source.archive_root / "archive_001.zip"
            first_stat = first_archive.stat()
            second_stat = second_archive.stat()
            registry = root / "registry.json"
            registry.write_text(
                json.dumps(
                    {
                        "schema_version": "mmprism.csl_news_source_integrity_registry.v1",
                        "source": {
                            "source_id": "fixture",
                            "source_revision": "revision",
                            "labels_sha256": "c" * 64,
                        },
                        "summary": {
                            "passed_count": 1,
                            "failed_count": 1,
                            "pending_count": 0,
                            "missing_count": 0,
                        },
                        "archives": {
                            "001": {
                                "archive_id": "001",
                                "archive_name": "archive_001.zip",
                                "status": "passed",
                                "source_present": True,
                                "size_bytes": first_stat.st_size,
                                "mtime_ns": first_stat.st_mtime_ns,
                                "sha256": "a" * 64,
                                "video_count": 2,
                            },
                            "002": {
                                "archive_id": "002",
                                "archive_name": "archive_002.zip",
                                "status": "failed",
                                "source_present": True,
                                "size_bytes": second_stat.st_size,
                                "mtime_ns": second_stat.st_mtime_ns,
                                "sha256": "b" * 64,
                                "video_count": 1,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            eligible_sidecar = (
                config.runtime.output_root
                / "samples"
                / "archive_001"
                / "sample.json"
            )
            eligible_payload = json.loads(
                eligible_sidecar.read_text(encoding="utf-8")
            )
            eligible_payload["source"] = {
                "integrity": {
                    "archive_sha256": "a" * 64,
                    "labels_sha256": "c" * 64,
                }
            }
            eligible_sidecar.write_text(
                json.dumps(eligible_payload), encoding="utf-8"
            )

            report = build_csl_news_annotation_status(
                config,
                sample_validate_count=1,
                integrity_registry_path=registry,
            )

        self.assertEqual(report["source"]["complete_archive_count"], 1)
        self.assertEqual(report["source"]["available_video_count"], 2)
        self.assertEqual(report["annotation"]["completed_sample_count"], 1)
        self.assertEqual(report["annotation"]["ineligible_npz_count"], 1)
        self.assertEqual(report["annotation"]["ineligible_sidecar_count"], 1)
        self.assertEqual(report["status"], "attention_required")


if __name__ == "__main__":
    unittest.main()
