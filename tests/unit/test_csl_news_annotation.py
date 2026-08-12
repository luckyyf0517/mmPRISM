import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mmprism.data import (
    CslNewsAnnotationError,
    canonicalize_hands,
    is_completed_annotation_archive,
    is_completed_annotation_sample,
    load_csl_news_annotation_config,
    run_csl_news_annotation,
    stable_sample_id,
    validate_annotation_output,
)
from mmprism.data.csl_news_annotation import (
    CslNewsAnnotationArtifactConflictError,
    _archive_integrity_provenance,
    _resolve_annotation_artifact_target,
    _source_variant_artifact_paths,
    _write_npz_atomic,
    discover_complete_archives,
)


class CslNewsAnnotationTest(unittest.TestCase):
    def _write_config(self, root: Path, *, extra_runtime: str = "") -> Path:
        config = root / "annotation.yaml"
        config.write_text(
            f"""schema_version: mmprism.csl_news_pose_annotation.v1
source:
  archive_root: archives
  labels_path: labels.json
  source_id: fixture
  source_revision: revision
  expected_archive_count: 2
model:
  mmpose_root: mmpose
  mmpose_commit: commit
  project_dir: project
  config_path: model.py
  checkpoint_path: model.pth
  checkpoint_sha256: {'a' * 64}
  device: cuda:0
transform:
  crop_top: 20
  crop_left: 20
  crop_right: 20
  confidence_threshold: 0.3
runtime:
  output_root: output
  scratch_root: scratch
  worker_index: 0
  worker_count: 1
  poll_seconds: 60
  min_free_bytes: 0
  max_consecutive_oom: 2
{extra_runtime}""",
            encoding="utf-8",
        )
        return config

    def _valid_arrays(self, frame_count: int = 2) -> dict[str, np.ndarray]:
        return {
            "native_keypoints_3d": np.zeros((frame_count, 133, 3), dtype=np.float32),
            "native_keypoint_scores": np.ones((frame_count, 133), dtype=np.float32),
            "transformed_keypoints_2d": np.zeros((frame_count, 133, 2), dtype=np.float32),
            "frame_indices": np.arange(frame_count, dtype=np.int64),
            "timestamps_seconds": np.arange(frame_count, dtype=np.float64) / 25,
            "canonical_pose": np.zeros((frame_count, 2, 24, 3), dtype=np.float32),
            "canonical_confidence": np.ones((frame_count, 2, 24), dtype=np.float32),
            "canonical_valid": np.ones((frame_count, 2, 24), dtype=np.bool_),
        }

    def _write_integrity_registry(
        self,
        root: Path,
        archive_stats: dict[int, tuple[int, int]],
        *,
        archive_paths: dict[int, str] | None = None,
    ) -> Path:
        archives = {}
        for archive_id, (size_bytes, mtime_ns) in archive_stats.items():
            key = f"{archive_id:03d}"
            archives[key] = {
                "archive_id": key,
                "archive_name": f"archive_{key}.zip",
                "status": "passed" if archive_id != 1 else "failed",
                "source_present": True,
                "size_bytes": size_bytes,
                "mtime_ns": mtime_ns,
                "sha256": "b" * 64,
                "video_count": archive_id,
                "audited_at": "2026-08-11T16:00:00+00:00",
                "builder_commit": "c" * 40,
                "audit": {
                    "path": f"manifests/audit_{key}.json",
                    "sha256": "d" * 64,
                },
            }
            if archive_paths is not None:
                relative_path = archive_paths[archive_id]
                archives[key]["archive_path_relative"] = relative_path
                archives[key]["source_kind"] = (
                    "primary"
                    if relative_path == f"archive_{key}.zip"
                    else "replacement"
                )
        path = root / "registry.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": (
                        "mmprism.csl_news_source_integrity_registry.v2"
                        if archive_paths is not None
                        else "mmprism.csl_news_source_integrity_registry.v1"
                    ),
                    "source": {
                        "source_id": "fixture",
                        "source_revision": "revision",
                        "labels_sha256": "e" * 64,
                    },
                    "archives": archives,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_loads_strict_config_and_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_csl_news_annotation_config(self._write_config(root), root)

        self.assertEqual(config.runtime.worker_count, 1)
        self.assertEqual(config.transform.crop_top, 20)
        self.assertEqual(config.source.archive_root, root / "archives")
        self.assertEqual(len(config.fingerprint), 64)

    def test_rejects_unknown_config_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_config(root, extra_runtime="  surprise: true\n")
            with self.assertRaisesRegex(CslNewsAnnotationError, "Unknown keys"):
                load_csl_news_annotation_config(path, root)

    def test_stable_sample_id_uses_full_source_identity(self) -> None:
        first = stable_sample_id("source", "archive_001.zip", "video.mp4")
        repeated = stable_sample_id("source", "archive_001.zip", "video.mp4")
        other_archive = stable_sample_id("source", "archive_002.zip", "video.mp4")

        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other_archive)
        self.assertEqual(len(first), 24)

    def test_canonical_mapping_preserves_native_and_centers_depth(self) -> None:
        native = np.arange(2 * 133 * 3, dtype=np.float32).reshape(2, 133, 3)
        original = native.copy()
        scores = np.linspace(0, 1, 2 * 133, dtype=np.float32).reshape(2, 133)
        canonical, confidence, valid, depth_center = canonicalize_hands(
            native, scores, 0.3
        )

        self.assertTrue(np.array_equal(native, original))
        self.assertEqual(canonical.shape, (2, 2, 24, 3))
        self.assertEqual(confidence.shape, (2, 2, 24))
        self.assertEqual(valid.shape, (2, 2, 24))
        self.assertAlmostEqual(depth_center, float(original[:, [6, 7], 2].mean()))
        np.testing.assert_array_equal(canonical[:, 0, 0, :2], original[:, 5, :2])
        np.testing.assert_array_equal(canonical[:, 1, 3, :2], original[:, 112, :2])

    def test_validates_output_and_resume_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "sample.npz"
            np.savez_compressed(artifact, **self._valid_arrays())
            checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
            sidecar = root / "sample.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "config_fingerprint": "fingerprint",
                        "artifact": {
                            "sha256": checksum,
                            "size_bytes": artifact.stat().st_size,
                        },
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(validate_annotation_output(artifact))
            self.assertTrue(
                is_completed_annotation_sample(artifact, sidecar, "fingerprint")
            )
            self.assertFalse(is_completed_annotation_sample(artifact, sidecar, "other"))

            invalid = self._valid_arrays()
            invalid["frame_indices"] = np.asarray([1, 2], dtype=np.int64)
            np.savez_compressed(artifact, **invalid)
            self.assertFalse(validate_annotation_output(artifact))

    def test_atomic_npz_writer_returns_durable_identity_and_refuses_overwrite(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "sample.npz"
            size_bytes, checksum = _write_npz_atomic(
                self._valid_arrays(), artifact
            )

            self.assertGreater(size_bytes, 0)
            self.assertEqual(size_bytes, artifact.stat().st_size)
            self.assertEqual(
                checksum, hashlib.sha256(artifact.read_bytes()).hexdigest()
            )
            self.assertTrue(validate_annotation_output(artifact))
            original = artifact.read_bytes()

            with self.assertRaisesRegex(
                CslNewsAnnotationArtifactConflictError, "refusing to overwrite"
            ):
                _write_npz_atomic(self._valid_arrays(frame_count=3), artifact)
            self.assertEqual(artifact.read_bytes(), original)

    def test_resume_requires_sidecar_size_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "sample.npz"
            size_bytes, checksum = _write_npz_atomic(
                self._valid_arrays(), artifact
            )
            sidecar = root / "sample.json"
            payload = {
                "status": "completed",
                "config_fingerprint": "fingerprint",
                "artifact": {"sha256": checksum, "size_bytes": size_bytes + 1},
            }
            sidecar.write_text(json.dumps(payload), encoding="utf-8")

            self.assertFalse(
                is_completed_annotation_sample(artifact, sidecar, "fingerprint")
            )
            payload["artifact"]["size_bytes"] = size_bytes
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(
                is_completed_annotation_sample(artifact, sidecar, "fingerprint")
            )

    def test_resume_routes_unbound_or_invalid_canonical_output_to_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_csl_news_annotation_config(self._write_config(root), root)
            sample_root = config.runtime.output_root / "samples" / "archive_005"
            sample_root.mkdir(parents=True)
            artifact = sample_root / "sample.npz"
            size_bytes, checksum = _write_npz_atomic(
                self._valid_arrays(), artifact
            )
            sidecar = sample_root / "sample.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "config_fingerprint": "fingerprint",
                        "source": {
                            "archive_size_bytes": 10,
                            "member_size_bytes": 20,
                            "member_crc32": 30,
                            "integrity": None,
                        },
                        "artifact": {
                            "sha256": checksum,
                            "size_bytes": size_bytes,
                        },
                    }
                ),
                encoding="utf-8",
            )
            current = {
                "archive_sha256": "b" * 64,
                "labels_sha256": "c" * 64,
            }

            self.assertFalse(
                is_completed_annotation_sample(
                    artifact,
                    sidecar,
                    "fingerprint",
                    archive_size_bytes=10,
                    member_size_bytes=20,
                    member_crc32=30,
                    source_integrity=current,
                )
            )
            target = _resolve_annotation_artifact_target(
                config,
                "archive_005.zip",
                "sample",
                "fingerprint",
                archive_size_bytes=10,
                member_size_bytes=20,
                member_crc32=30,
                source_integrity=current,
            )
            self.assertFalse(target.completed)
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            payload["source"]["integrity"] = current
            payload["artifact"]["size_bytes"] = size_bytes + 1
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            invalid_same_source_target = _resolve_annotation_artifact_target(
                config,
                "archive_005.zip",
                "sample",
                "fingerprint",
                archive_size_bytes=10,
                member_size_bytes=20,
                member_crc32=30,
                source_integrity=current,
            )
            variant_npz, variant_sidecar = _source_variant_artifact_paths(
                config, "archive_005.zip", "sample", current
            )
            variant_size, variant_checksum = _write_npz_atomic(
                self._valid_arrays(), variant_npz
            )
            payload["artifact"] = {
                "size_bytes": variant_size,
                "sha256": variant_checksum,
            }
            variant_sidecar.write_text(json.dumps(payload), encoding="utf-8")
            resumed_target = _resolve_annotation_artifact_target(
                config,
                "archive_005.zip",
                "sample",
                "fingerprint",
                archive_size_bytes=10,
                member_size_bytes=20,
                member_crc32=30,
                source_integrity=current,
            )

        self.assertEqual(target.npz_path, variant_npz)
        self.assertEqual(target.sidecar_path, variant_sidecar)
        self.assertEqual(invalid_same_source_target.npz_path, variant_npz)
        self.assertEqual(invalid_same_source_target.sidecar_path, variant_sidecar)
        self.assertTrue(resumed_target.completed)
        self.assertEqual(resumed_target.npz_path, variant_npz)
        self.assertEqual(
            variant_npz.name, f"sample--source_{'b' * 64}.npz"
        )
        self.assertEqual(
            variant_sidecar.name, f"sample--source_{'b' * 64}.json"
        )

    def test_retries_archive_markers_with_unresolved_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "archive_005.json"
            marker.write_text(
                json.dumps(
                    {
                        "status": "completed_with_failures",
                        "config_fingerprint": "fingerprint",
                        "archive_size_bytes": 1024,
                        "failed": 2,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(
                is_completed_annotation_archive(marker, "fingerprint", 1024)
            )

            marker.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "config_fingerprint": "fingerprint",
                        "archive_size_bytes": 1024,
                        "failed": 0,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(
                is_completed_annotation_archive(marker, "fingerprint", 1024)
            )
            self.assertFalse(
                is_completed_annotation_archive(marker, "fingerprint", 2048)
            )

    def test_registry_filters_failed_archives_and_applies_worker_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_csl_news_annotation_config(self._write_config(root), root)
            config.source.archive_root.mkdir()
            stats = {}
            for archive_id in (1, 2):
                path = config.source.archive_root / f"archive_{archive_id:03d}.zip"
                path.write_bytes(f"archive-{archive_id}".encode())
                stat = path.stat()
                stats[archive_id] = (stat.st_size, stat.st_mtime_ns)
            registry = self._write_integrity_registry(root, stats)

            archives = discover_complete_archives(
                config,
                worker_index=0,
                worker_count=2,
                integrity_registry_path=registry,
            )

        self.assertEqual([path.name for path in archives], ["archive_002.zip"])

    def test_registry_rejects_archive_changed_after_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_csl_news_annotation_config(self._write_config(root), root)
            config.source.archive_root.mkdir()
            archive = config.source.archive_root / "archive_002.zip"
            archive.write_bytes(b"original")
            stat = archive.stat()
            registry = self._write_integrity_registry(
                root, {2: (stat.st_size, stat.st_mtime_ns)}
            )
            archive.write_bytes(b"changed-source")

            with self.assertRaisesRegex(CslNewsAnnotationError, "changed after audit"):
                discover_complete_archives(
                    config, integrity_registry_path=registry
                )

    def test_registry_resolves_audited_replacement_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_csl_news_annotation_config(self._write_config(root), root)
            replacement_relative = (
                "replacements/recovery_v1/rgb_archives/archive_002.zip"
            )
            replacement = config.source.archive_root / replacement_relative
            replacement.parent.mkdir(parents=True)
            replacement.write_bytes(b"replacement")
            stat = replacement.stat()
            registry = self._write_integrity_registry(
                root,
                {2: (stat.st_size, stat.st_mtime_ns)},
                archive_paths={2: replacement_relative},
            )

            archives = discover_complete_archives(
                config, integrity_registry_path=registry
            )
            provenance = _archive_integrity_provenance(
                config, registry, replacement
            )

        self.assertEqual(archives, [replacement.resolve()])
        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertEqual(provenance["archive_path_relative"], replacement_relative)
        self.assertEqual(provenance["archive_source_kind"], "replacement")

    def test_archive_provenance_binds_one_registry_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_csl_news_annotation_config(self._write_config(root), root)
            config.source.archive_root.mkdir()
            archive = config.source.archive_root / "archive_002.zip"
            archive.write_bytes(b"source")
            stat = archive.stat()
            registry = self._write_integrity_registry(
                root, {2: (stat.st_size, stat.st_mtime_ns)}
            )
            expected_registry_sha256 = hashlib.sha256(registry.read_bytes()).hexdigest()

            provenance = _archive_integrity_provenance(config, registry, archive)

        self.assertIsNotNone(provenance)
        assert provenance is not None
        self.assertEqual(
            provenance["registry_sha256"],
            expected_registry_sha256,
        )
        self.assertEqual(provenance["archive_sha256"], "b" * 64)
        self.assertEqual(provenance["audit_sha256"], "d" * 64)
        self.assertEqual(provenance["builder_commit"], "c" * 40)

    def test_cooperative_pause_stops_before_the_next_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = load_csl_news_annotation_config(self._write_config(root), root)
            config.source.archive_root.mkdir()
            config.source.labels_path.write_text(
                json.dumps([{"video": "sample.mp4", "text": "测试文本"}]),
                encoding="utf-8",
            )
            with zipfile.ZipFile(config.source.archive_root / "archive_001.zip", "w") as archive:
                archive.writestr("sample.mp4", b"video")

            with patch(
                "mmprism.data.csl_news_annotation._require_model_assets", return_value={}
            ):
                result = run_csl_news_annotation(
                    config,
                    archive_id=1,
                    continue_requested=lambda: False,
                )

            output_root_exists = (config.runtime.output_root / "samples").exists()

        self.assertEqual(result["status"], "paused")
        self.assertEqual(result["processed"], 0)
        self.assertEqual(result["skipped"], 0)
        self.assertFalse(output_root_exists)


if __name__ == "__main__":
    unittest.main()
