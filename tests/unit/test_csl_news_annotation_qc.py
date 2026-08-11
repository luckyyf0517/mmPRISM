import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mmprism.data import (
    CslNewsAnnotationConfig,
    build_csl_news_annotation_qc,
    load_csl_news_annotation_config,
    write_csl_news_annotation_qc,
)


class CslNewsAnnotationQcTest(unittest.TestCase):
    def _config(self, root: Path) -> CslNewsAnnotationConfig:
        path = root / "qc.yaml"
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

    def _write_sample(
        self,
        config: CslNewsAnnotationConfig,
        *,
        reported_frames: int = 2,
        transformed_value: float = 5.0,
    ) -> Path:
        sample_root = config.runtime.output_root / "samples" / "archive_001"
        sample_root.mkdir(parents=True)
        artifact = sample_root / "sample.npz"
        frame_count = 2
        np.savez_compressed(
            artifact,
            native_keypoints_3d=np.ones((frame_count, 133, 3), dtype=np.float32),
            native_keypoint_scores=np.full((frame_count, 133), 0.9, dtype=np.float32),
            transformed_keypoints_2d=np.full(
                (frame_count, 133, 2), transformed_value, dtype=np.float32
            ),
            frame_indices=np.arange(frame_count, dtype=np.int64),
            timestamps_seconds=np.arange(frame_count, dtype=np.float64) / 25,
            canonical_pose=np.ones((frame_count, 2, 24, 3), dtype=np.float32),
            canonical_confidence=np.full(
                (frame_count, 2, 24), 0.9, dtype=np.float32
            ),
            canonical_valid=np.ones((frame_count, 2, 24), dtype=np.bool_),
        )
        checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
        sidecar = sample_root / "sample.json"
        sidecar.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "sample_id": "sample",
                    "config_fingerprint": config.fingerprint,
                    "annotation": {"text": "测试文本"},
                    "video": {
                        "reported_frame_count": reported_frames,
                        "fps": 25.0,
                        "cropped_width": 10,
                        "cropped_height": 10,
                    },
                    "artifact": {"sha256": checksum},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return sidecar

    def test_passes_valid_pose_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            self._write_sample(config)
            report = build_csl_news_annotation_qc(config, sample_count=1)

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["summary"]["failure_sample_count"], 0)
        self.assertEqual(report["summary"]["warning_sample_count"], 0)
        self.assertEqual(report["summary"]["total_frame_count"], 2)
        self.assertEqual(report["summary"]["canonical_valid_ratio"], 1.0)

    def test_fails_frame_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            self._write_sample(config, reported_frames=3)
            report = build_csl_news_annotation_qc(config, sample_count=1)

        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "decoded and reported frame counts differ",
            report["samples"][0]["failures"],
        )

    def test_warns_for_out_of_bounds_keypoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self._config(Path(directory))
            self._write_sample(config, transformed_value=20.0)
            report = build_csl_news_annotation_qc(config, sample_count=1)

        self.assertEqual(report["status"], "passed_with_warnings")
        self.assertEqual(report["summary"]["warning_sample_count"], 1)

    def test_writes_qc_report_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "qc" / "report.json"
            written = write_csl_news_annotation_qc({"status": "passed"}, output)
            payload = json.loads(written.read_text(encoding="utf-8"))

        self.assertEqual(payload, {"status": "passed"})


if __name__ == "__main__":
    unittest.main()
