import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mmprism.data import (
    CslNewsAnnotationError,
    canonicalize_hands,
    is_completed_annotation_sample,
    load_csl_news_annotation_config,
    stable_sample_id,
    validate_annotation_output,
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
                        "artifact": {"sha256": checksum},
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


if __name__ == "__main__":
    unittest.main()
