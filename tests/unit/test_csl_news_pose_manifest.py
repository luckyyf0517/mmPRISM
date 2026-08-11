import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from mmprism.contracts import validate_manifest
from mmprism.data import (
    CslNewsPoseManifest,
    CslNewsPoseManifestConfig,
    CslNewsPoseManifestError,
    build_csl_news_pose_manifest_snapshot,
    load_csl_news_pose_manifest_config,
    stable_sample_id,
)


class CslNewsPoseManifestTest(unittest.TestCase):
    def _config(self, root: Path) -> CslNewsPoseManifestConfig:
        config_path = root / "pose_manifest.yaml"
        config_path.write_text(
            f"""schema_version: mmprism.csl_news_pose_manifest.v1
source:
  data_root: ${{MMPRISM_TEST_DATA_ROOT}}
  labels_path: incoming/metadata/labels.json
  integrity_registry: manifests/integrity/registry.json
  source_id: fixture:csl-news
  source_revision: revision
  expected_archive_count: 2
annotation:
  root: interim/pose/v1
  dataset_id: fixture_csl_news_pose
  config_fingerprint: {'a' * 64}
validation:
  verify_artifact_checksum: true
  validate_artifact_contract: true
  minimum_free_bytes: 0
output:
  snapshot_root: manifests/pose/v1
""",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"MMPRISM_TEST_DATA_ROOT": str(root)}):
            return load_csl_news_pose_manifest_config(config_path)

    def _arrays(self, frame_count: int = 2) -> dict[str, np.ndarray]:
        return {
            "native_keypoints_3d": np.zeros((frame_count, 133, 3), dtype=np.float32),
            "native_keypoint_scores": np.ones((frame_count, 133), dtype=np.float32),
            "transformed_keypoints_2d": np.zeros(
                (frame_count, 133, 2), dtype=np.float32
            ),
            "frame_indices": np.arange(frame_count, dtype=np.int64),
            "timestamps_seconds": np.arange(frame_count, dtype=np.float64) / 25,
            "canonical_pose": np.zeros((frame_count, 2, 24, 3), dtype=np.float32),
            "canonical_confidence": np.ones(
                (frame_count, 2, 24), dtype=np.float32
            ),
            "canonical_valid": np.ones((frame_count, 2, 24), dtype=np.bool_),
        }

    def _write_sample(
        self,
        config: CslNewsPoseManifestConfig,
        *,
        archive_id: int,
        video_name: str,
        caption: str,
    ) -> tuple[Path, Path]:
        archive_name = f"archive_{archive_id:03d}.zip"
        sample_id = stable_sample_id(config.source_id, archive_name, video_name)
        sample_root = (
            config.annotation_root / "samples" / f"archive_{archive_id:03d}"
        )
        sample_root.mkdir(parents=True, exist_ok=True)
        artifact_path = sample_root / f"{sample_id}.npz"
        arrays = self._arrays()
        np.savez_compressed(artifact_path, **arrays)
        artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        source_integrity = None
        if config.integrity_registry_path.is_file():
            registry = json.loads(
                config.integrity_registry_path.read_text(encoding="utf-8")
            )
            entry = registry.get("archives", {}).get(f"{archive_id:03d}", {})
            archive_sha256 = entry.get("sha256")
            labels_sha256 = registry.get("source", {}).get("labels_sha256")
            if isinstance(archive_sha256, str) and isinstance(labels_sha256, str):
                source_integrity = {
                    "archive_sha256": archive_sha256,
                    "labels_sha256": labels_sha256,
                }
        sidecar_path = artifact_path.with_suffix(".json")
        sidecar_path.write_text(
            json.dumps(
                {
                    "schema_version": "mmprism.csl_news_pose_sample.v1",
                    "status": "completed",
                    "sample_id": sample_id,
                    "config_fingerprint": "a" * 64,
                    "source": {
                        "source_id": config.source_id,
                        "source_revision": config.source_revision,
                        "archive": archive_name,
                        "archive_size_bytes": 100 + archive_id,
                        "member": video_name,
                        "member_size_bytes": 1234,
                        "member_crc32": archive_id,
                        "video_sha256": "b" * 64,
                        "integrity": source_integrity,
                    },
                    "annotation": {
                        "text": caption,
                        "legacy_pose_name": video_name.replace(".mp4", ".pkl"),
                    },
                    "arrays": {
                        name: list(array.shape)
                        for name, array in arrays.items()
                        if name
                        not in {
                            "frame_indices",
                            "timestamps_seconds",
                        }
                    },
                    "model": {
                        "mmpose_commit": "c" * 40,
                        "config_sha256": "d" * 64,
                        "checkpoint_sha256": "e" * 64,
                    },
                    "transform": {
                        "crop_top": 20,
                        "crop_left": 20,
                        "crop_right": 20,
                        "confidence_threshold": 0.3,
                    },
                    "artifact": {
                        "path": str(artifact_path),
                        "size_bytes": artifact_path.stat().st_size,
                        "sha256": artifact_sha256,
                    },
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return artifact_path, sidecar_path

    def _prepare(self, root: Path) -> tuple[CslNewsPoseManifestConfig, bytes]:
        config = self._config(root)
        labels_path = config.labels_path
        labels_path.parent.mkdir(parents=True)
        labels = [
            {
                "video": "Common-Concerns_20200101_0-100_1.mp4",
                "pose": "Common-Concerns_20200101_0-100_1.pkl",
                "text": "第一条文本",
            },
            {
                "video": "20200102_Dragon-TV__0-100_2.mp4",
                "pose": "20200102_Dragon-TV__0-100_2.pkl",
                "text": "隔离文本",
            },
        ]
        labels_path.write_text(
            json.dumps(labels, ensure_ascii=False), encoding="utf-8"
        )
        labels_sha256 = hashlib.sha256(labels_path.read_bytes()).hexdigest()
        registry_path = config.integrity_registry_path
        registry_path.parent.mkdir(parents=True)
        registry = {
            "schema_version": "mmprism.csl_news_source_integrity_registry.v1",
            "source": {
                "source_id": config.source_id,
                "source_revision": config.source_revision,
                "expected_archive_count": 2,
                "labels_sha256": labels_sha256,
            },
            "summary": {"passed_count": 1, "failed_count": 1},
            "archives": {
                "001": {
                    "archive_id": "001",
                    "archive_name": "archive_001.zip",
                    "status": "passed",
                    "source_present": True,
                    "size_bytes": 101,
                    "mtime_ns": 1001,
                    "sha256": "1" * 64,
                    "video_count": 1,
                    "audited_at": "2026-08-11T16:00:00+00:00",
                    "builder_commit": "2" * 40,
                    "audit": {
                        "path": "manifests/integrity/audit_001.json",
                        "sha256": "3" * 64,
                    },
                },
                "002": {
                    "archive_id": "002",
                    "archive_name": "archive_002.zip",
                    "status": "failed",
                    "source_present": True,
                    "size_bytes": 102,
                    "mtime_ns": 1002,
                },
            },
        }
        registry_bytes = (
            json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        registry_path.write_bytes(registry_bytes)
        self._write_sample(
            config,
            archive_id=1,
            video_name=labels[0]["video"],
            caption=labels[0]["text"],
        )
        self._write_sample(
            config,
            archive_id=2,
            video_name=labels[1]["video"],
            caption=labels[1]["text"],
        )
        return config, registry_bytes

    def _runtime(self, *, dirty: bool = False) -> dict[str, object]:
        return {"git": {"commit": "f" * 40, "dirty": dirty}, "python": "3.12"}

    def _prepare_exclusion(
        self, root: Path
    ) -> tuple[CslNewsPoseManifestConfig, Path, Path, Path]:
        config, _ = self._prepare(root)
        labels = json.loads(config.labels_path.read_text(encoding="utf-8"))
        second_video = "Common-Concerns_20200103_0-100_3.mp4"
        second_caption = "保留文本"
        labels.append(
            {
                "video": second_video,
                "pose": second_video.replace(".mp4", ".pkl"),
                "text": second_caption,
            }
        )
        config.labels_path.write_text(
            json.dumps(labels, ensure_ascii=False), encoding="utf-8"
        )
        labels_sha256 = hashlib.sha256(config.labels_path.read_bytes()).hexdigest()
        registry = json.loads(
            config.integrity_registry_path.read_text(encoding="utf-8")
        )
        registry["source"]["labels_sha256"] = labels_sha256
        registry["archives"]["001"]["video_count"] = 2
        config.integrity_registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for sidecar_path in (
            config.annotation_root / "samples" / "archive_001"
        ).glob("*.json"):
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            payload["source"]["integrity"]["labels_sha256"] = labels_sha256
            sidecar_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        self._write_sample(
            config,
            archive_id=1,
            video_name=second_video,
            caption=second_caption,
        )

        bad_video = labels[0]["video"]
        bad_sample_id = stable_sample_id(
            config.source_id, "archive_001.zip", bad_video
        )
        bad_sidecar = (
            config.annotation_root
            / "samples"
            / "archive_001"
            / f"{bad_sample_id}.json"
        )
        bad_artifact = bad_sidecar.with_suffix(".npz")
        payload = json.loads(bad_sidecar.read_text(encoding="utf-8"))
        payload["artifact"]["size_bytes"] = 0
        payload["artifact"]["sha256"] = hashlib.sha256(b"").hexdigest()
        bad_sidecar.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sidecar_sha256 = hashlib.sha256(bad_sidecar.read_bytes()).hexdigest()
        artifact_sha256 = hashlib.sha256(bad_artifact.read_bytes()).hexdigest()
        sidecar_relative = bad_sidecar.relative_to(config.annotation_root).as_posix()
        artifact_relative = bad_artifact.relative_to(config.annotation_root).as_posix()
        report = {
            "schema_version": "mmprism.csl_news_pose_annotation_identity_audit.v1",
            "status": "failed",
            "runtime": {"git": {"commit": "e" * 40, "dirty": False}},
            "scope": {"config_fingerprint": "a" * 64},
            "invalid_pairs": [
                {
                    "sample_id": bad_sample_id,
                    "sidecar": sidecar_relative,
                    "artifact": artifact_relative,
                    "failures": [
                        "artifact_sha256_mismatch",
                        "artifact_size_mismatch",
                    ],
                    "sidecar_identity": {"sha256": sidecar_sha256},
                    "observed_artifact": {
                        "size_bytes": bad_artifact.stat().st_size,
                        "sha256": artifact_sha256,
                    },
                    "declared_artifact": {
                        "size_bytes": 0,
                        "sha256": hashlib.sha256(b"").hexdigest(),
                    },
                }
            ],
        }
        report_path = config.annotation_root / "identity_audits" / "audit_fixture.json"
        report_path.parent.mkdir(parents=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()

        config_path = root / "pose_manifest.yaml"
        config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config_payload["validation"]["exclusions"] = [
            {
                "sample_id": bad_sample_id,
                "archive_directory": "archive_001",
                "sidecar_sha256": sidecar_sha256,
                "artifact_size_bytes": bad_artifact.stat().st_size,
                "artifact_sha256": artifact_sha256,
                "declared_artifact_size_bytes": 0,
                "declared_artifact_sha256": hashlib.sha256(b"").hexdigest(),
                "audit_report": "identity_audits/audit_fixture.json",
                "audit_report_sha256": report_sha256,
                "reason": "sidecar_artifact_identity_conflict",
            }
        ]
        config_path.write_text(
            yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8"
        )
        with patch.dict(os.environ, {"MMPRISM_TEST_DATA_ROOT": str(root)}):
            excluded_config = load_csl_news_pose_manifest_config(config_path)
        return excluded_config, bad_artifact, bad_sidecar, report_path

    def test_builds_portable_snapshot_and_adapter_loads_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, registry_bytes = self._prepare(root)
            receipt = build_csl_news_pose_manifest_snapshot(
                config,
                runtime_report=self._runtime(),
                snapshot_id="fixture",
            )
            manifest_path = Path(receipt["manifest_path"])
            manifest_text = manifest_path.read_text(encoding="utf-8")
            record = json.loads(manifest_text)
            summary = json.loads(
                Path(receipt["summary_path"]).read_text(encoding="utf-8")
            )
            contract = validate_manifest(manifest_path)
            dataset = CslNewsPoseManifest(
                manifest_path, config.annotation_root, verify_checksum=True
            )
            sample = dataset[0]
            copied_registry = (
                Path(receipt["snapshot_dir"]) / "integrity_registry.json"
            ).read_bytes()

        self.assertEqual(receipt["status"], "partial")
        self.assertEqual(receipt["record_count"], 1)
        self.assertEqual(contract.datasets, ("fixture_csl_news_pose",))
        self.assertIn("canonical_pose", contract.modalities)
        self.assertNotIn(str(root), manifest_text)
        self.assertEqual(record["modalities"]["caption"]["text"], "第一条文本")
        self.assertEqual(record["modalities"]["canonical_pose"]["shape"], [2, 2, 24, 3])
        self.assertEqual(summary["annotation"]["ineligible_sidecar_count"], 1)
        self.assertEqual(summary["annotation"]["ineligible_npz_count"], 1)
        self.assertEqual(summary["annotation"]["sidecar_integrity_missing_count"], 0)
        self.assertEqual(copied_registry, registry_bytes)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(sample.caption, "第一条文本")
        self.assertEqual(sample.arrays["native_keypoints_3d"].shape, (2, 133, 3))
        self.assertEqual(sample.arrays["canonical_pose"].shape, (2, 2, 24, 3))

    def test_rejects_dirty_git_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self._prepare(Path(directory))
            with self.assertRaisesRegex(CslNewsPoseManifestError, "clean Git"):
                build_csl_news_pose_manifest_snapshot(
                    config,
                    runtime_report=self._runtime(dirty=True),
                    snapshot_id="dirty",
                )

    def test_selects_current_source_variant_and_quarantines_unbound_sidecar(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self._prepare(root)
            sample_root = config.annotation_root / "samples" / "archive_001"
            canonical_sidecar = next(sample_root.glob("*.json"))
            canonical_artifact = canonical_sidecar.with_suffix(".npz")
            payload = json.loads(canonical_sidecar.read_text(encoding="utf-8"))
            source_integrity = dict(payload["source"]["integrity"])
            payload["source"]["integrity"] = None
            canonical_sidecar.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )

            sample_id = payload["sample_id"]
            archive_sha256 = source_integrity["archive_sha256"]
            variant_stem = f"{sample_id}--source_{archive_sha256}"
            variant_artifact = sample_root / f"{variant_stem}.npz"
            variant_artifact.write_bytes(canonical_artifact.read_bytes())
            variant_sidecar = sample_root / f"{variant_stem}.json"
            payload["source"]["integrity"] = source_integrity
            payload["artifact"]["path"] = str(variant_artifact)
            payload["artifact"]["size_bytes"] = variant_artifact.stat().st_size
            payload["artifact"]["sha256"] = hashlib.sha256(
                variant_artifact.read_bytes()
            ).hexdigest()
            variant_sidecar.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )

            receipt = build_csl_news_pose_manifest_snapshot(
                config,
                runtime_report=self._runtime(),
                snapshot_id="source_variant",
            )
            snapshot = Path(receipt["snapshot_dir"])
            summary = json.loads(
                Path(receipt["summary_path"]).read_text(encoding="utf-8")
            )
            manifest_record = json.loads(
                Path(receipt["manifest_path"]).read_text(encoding="utf-8")
            )
            quarantine_lines = (
                snapshot / "source_identity_quarantine.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            checksum_text = (snapshot / "SHA256SUMS").read_text(encoding="ascii")

        self.assertEqual(receipt["record_count"], 1)
        self.assertEqual(
            summary["annotation"]["source_identity_quarantined_sidecar_count"],
            1,
        )
        self.assertEqual(len(quarantine_lines), 1)
        self.assertIn(variant_stem, manifest_record["modalities"]["canonical_pose"]["uri"])
        self.assertIn("source_identity_quarantine.jsonl", checksum_text)

    def test_rejects_duplicate_current_source_sidecars_without_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self._prepare(root)
            sample_root = config.annotation_root / "samples" / "archive_001"
            canonical_sidecar = next(sample_root.glob("*.json"))
            canonical_artifact = canonical_sidecar.with_suffix(".npz")
            payload = json.loads(canonical_sidecar.read_text(encoding="utf-8"))
            sample_id = payload["sample_id"]
            archive_sha256 = payload["source"]["integrity"]["archive_sha256"]
            variant_stem = f"{sample_id}--source_{archive_sha256}"
            variant_artifact = sample_root / f"{variant_stem}.npz"
            variant_artifact.write_bytes(canonical_artifact.read_bytes())
            payload["artifact"] = {
                "path": str(variant_artifact),
                "size_bytes": variant_artifact.stat().st_size,
                "sha256": hashlib.sha256(variant_artifact.read_bytes()).hexdigest(),
            }
            (sample_root / f"{variant_stem}.json").write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )

            with self.assertRaisesRegex(
                CslNewsPoseManifestError,
                "multiple sidecars bind the current source identity",
            ):
                build_csl_news_pose_manifest_snapshot(
                    config,
                    runtime_report=self._runtime(),
                    snapshot_id="unapproved-duplicate",
                )

    def test_rejects_artifact_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self._prepare(root)
            eligible_artifact = next(
                (config.annotation_root / "samples" / "archive_001").glob("*.npz")
            )
            np.savez_compressed(eligible_artifact, **self._arrays(frame_count=3))
            with self.assertRaisesRegex(CslNewsPoseManifestError, "checksum mismatch"):
                build_csl_news_pose_manifest_snapshot(
                    config,
                    runtime_report=self._runtime(),
                    snapshot_id="checksum",
                )

    def test_applies_only_checksum_bound_exclusion_and_copies_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, bad_artifact, bad_sidecar, report_path = self._prepare_exclusion(
                Path(directory)
            )
            artifact_before = bad_artifact.read_bytes()
            sidecar_before = bad_sidecar.read_bytes()
            receipt = build_csl_news_pose_manifest_snapshot(
                config,
                runtime_report=self._runtime(),
                snapshot_id="excluded",
            )
            snapshot = Path(receipt["snapshot_dir"])
            summary = json.loads(
                Path(receipt["summary_path"]).read_text(encoding="utf-8")
            )
            exclusion = summary["annotation"]["exclusions"][0]
            evidence_copy = snapshot / exclusion["audit_report_snapshot_path"]
            checksum_text = (snapshot / "SHA256SUMS").read_text(encoding="ascii")

            self.assertEqual(receipt["record_count"], 1)
            self.assertEqual(summary["annotation"]["excluded_sidecar_count"], 1)
            self.assertEqual(summary["annotation"]["included_sidecar_count"], 1)
            self.assertEqual(evidence_copy.read_bytes(), report_path.read_bytes())
            self.assertIn(exclusion["audit_report_snapshot_path"], checksum_text)
            self.assertEqual(bad_artifact.read_bytes(), artifact_before)
            self.assertEqual(bad_sidecar.read_bytes(), sidecar_before)

    def test_excluded_canonical_pair_can_coexist_with_valid_recovery_variant(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, bad_artifact, bad_sidecar, _ = self._prepare_exclusion(
                Path(directory)
            )
            artifact_before = bad_artifact.read_bytes()
            sidecar_before = bad_sidecar.read_bytes()
            payload = json.loads(bad_sidecar.read_text(encoding="utf-8"))
            sample_id = payload["sample_id"]
            archive_sha256 = payload["source"]["integrity"]["archive_sha256"]
            variant_stem = f"{sample_id}--source_{archive_sha256}"
            variant_artifact = bad_artifact.with_name(f"{variant_stem}.npz")
            np.savez_compressed(variant_artifact, **self._arrays(frame_count=3))
            variant_sidecar = bad_sidecar.with_name(f"{variant_stem}.json")
            payload["artifact"] = {
                "path": str(variant_artifact),
                "variant": f"source_{archive_sha256}",
                "size_bytes": variant_artifact.stat().st_size,
                "sha256": hashlib.sha256(variant_artifact.read_bytes()).hexdigest(),
            }
            variant_sidecar.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )

            receipt = build_csl_news_pose_manifest_snapshot(
                config,
                runtime_report=self._runtime(),
                snapshot_id="recovery-variant",
            )
            records = [
                json.loads(line)
                for line in Path(receipt["manifest_path"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            summary = json.loads(
                Path(receipt["summary_path"]).read_text(encoding="utf-8")
            )

            self.assertEqual(receipt["record_count"], 2)
            self.assertEqual(summary["annotation"]["excluded_sidecar_count"], 1)
            recovered = next(
                record for record in records if record["sample_id"] == sample_id
            )
            self.assertIn(
                variant_stem,
                recovered["modalities"]["canonical_pose"]["uri"],
            )
            self.assertEqual(bad_artifact.read_bytes(), artifact_before)
            self.assertEqual(bad_sidecar.read_bytes(), sidecar_before)

    def test_exclusion_rejects_observed_artifact_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, bad_artifact, _, _ = self._prepare_exclusion(Path(directory))
            bad_artifact.write_bytes(bad_artifact.read_bytes() + b"drift")

            with self.assertRaisesRegex(
                CslNewsPoseManifestError, "observed identity mismatch"
            ):
                build_csl_news_pose_manifest_snapshot(
                    config,
                    runtime_report=self._runtime(),
                    snapshot_id="artifact-drift",
                )

    def test_exclusion_rejects_audit_report_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config, _, _, report_path = self._prepare_exclusion(Path(directory))
            report_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(
                CslNewsPoseManifestError, "report SHA-256 mismatch"
            ):
                build_csl_news_pose_manifest_snapshot(
                    config,
                    runtime_report=self._runtime(),
                    snapshot_id="report-drift",
                )

    def test_rejects_absolute_storage_relationship(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "invalid.yaml"
            path.write_text(
                f"""schema_version: mmprism.csl_news_pose_manifest.v1
source:
  data_root: {root}
  labels_path: labels.json
  integrity_registry: registry.json
  source_id: fixture
  source_revision: revision
  expected_archive_count: 1
annotation:
  root: /absolute/pose
  dataset_id: fixture
  config_fingerprint: {'a' * 64}
validation:
  verify_artifact_checksum: true
  validate_artifact_contract: true
  minimum_free_bytes: 0
output:
  snapshot_root: manifests
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CslNewsPoseManifestError, "must be relative"):
                load_csl_news_pose_manifest_config(path)

    def test_adapter_rejects_mixed_container_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self._prepare(root)
            receipt = build_csl_news_pose_manifest_snapshot(
                config,
                runtime_report=self._runtime(),
                snapshot_id="mixed-checksum",
            )
            manifest_path = Path(receipt["manifest_path"])
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            record["modalities"]["canonical_valid"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                CslNewsPoseManifestError, "share one NPZ checksum"
            ):
                CslNewsPoseManifest(manifest_path, config.annotation_root)


if __name__ == "__main__":
    unittest.main()
