import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mmprism.data import (
    CslDailyPoseAnnotationError,
    load_csl_daily_pose_annotation_config,
    run_csl_daily_pose_annotation,
)
from mmprism.data.csl_daily_pose_annotation import (
    ANNOTATION_V2_SCHEMA_VERSION,
    CslDailyPoseAnnotationConflictError,
    FramePosePrediction,
    build_sequence_annotation_v2,
    discover_sequences,
    list_sequence_frames,
    reduce_sequence_poses,
    subject_id_for_sequence,
    validate_annotation_v2_audit,
)
from mmprism.data.csl_daily_simulation_run import load_pose_manifest
from mmprism.data.csl_news_annotation import LEFT_JOINT_INDICES, RIGHT_JOINT_INDICES


class _FakeEstimator:
    """Scripted per-frame estimator keyed by sequence id."""

    def __init__(self, scripted: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self._scripted = scripted
        self.calls: list[Path] = []

    def estimate_frame(self, image_path: Path) -> FramePosePrediction:
        self.calls.append(image_path)
        sequence_id = image_path.parent.name
        keypoints, scores = self._scripted[sequence_id]
        index = sum(1 for call in self.calls if call.parent.name == sequence_id) - 1
        return FramePosePrediction(
            keypoints=keypoints[index].copy(), scores=scores[index].copy()
        )

    def runtime_metadata(self) -> dict[str, object]:
        return {"device": "fake", "inference_batch_size": 1}


def _full_score_keypoints(frame_count: int, z_base: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """[T, 133, 3] finite keypoints with deterministic coordinates."""
    keypoints = np.zeros((frame_count, 133, 3), dtype=np.float32)
    joints = np.arange(133, dtype=np.float32)
    for frame in range(frame_count):
        keypoints[frame, :, 0] = joints + frame * 1000.0
        keypoints[frame, :, 1] = joints * 2.0 + frame
        keypoints[frame, :, 2] = z_base + joints / 100.0 + frame
    scores = np.ones((frame_count, 133), dtype=np.float32)
    return keypoints, scores


class CslDailyPoseAnnotationTest(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config = root / "annotation.yaml"
        config.write_text(
            f"""schema_version: mmprism.csl_daily_pose_annotation.v1
source:
  sequence_root: ${{TEST_ROOT}}/sequences
  source_id: fixture_source
model:
  mmpose_root: mmpose
  mmpose_commit: {'c' * 40}
  project_dir: project
  config_path: model.py
  checkpoint_path: model.pth
  checkpoint_sha256: {'a' * 64}
  device: cuda:0
transform:
  confidence_threshold: 0.5
runtime:
  output_root: ${{TEST_ROOT}}/output
  inference_batch_size: 1
""",
            encoding="utf-8",
        )
        return config

    def _load_config(self, root: Path):
        return load_csl_daily_pose_annotation_config(
            self._write_config(root), root, variables={"TEST_ROOT": str(root)}
        )

    def _load_v2_config(self, root: Path):
        config = self._write_config(root)
        config.write_text(
            config.read_text(encoding="utf-8")
            .replace(
                "schema_version: mmprism.csl_daily_pose_annotation.v1",
                f"schema_version: {ANNOTATION_V2_SCHEMA_VERSION}",
            )
            .replace(
                "  confidence_threshold: 0.5\n",
                "  confidence_threshold: 0.5\n"
                "  minimum_valid_joints_per_frame: 12\n"
                "  minimum_valid_frame_ratio: 0.8\n",
            ),
            encoding="utf-8",
        )
        return load_csl_daily_pose_annotation_config(
            config, root, variables={"TEST_ROOT": str(root)}
        )

    def _write_sequence(
        self, root: Path, sequence_id: str, frame_names: list[str]
    ) -> Path:
        sequence_dir = root / "sequences" / sequence_id
        sequence_dir.mkdir(parents=True, exist_ok=True)
        for name in frame_names:
            (sequence_dir / name).write_bytes(b"fake-image-bytes")
        return sequence_dir

    def test_loads_config_and_expands_variables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._load_config(root)

        self.assertEqual(config.source.sequence_root, root / "sequences")
        self.assertEqual(config.runtime.output_root, root / "output")
        self.assertEqual(config.runtime.inference_batch_size, 1)
        self.assertEqual(config.transform.confidence_threshold, 0.5)
        self.assertEqual(config.model.mmpose_root, root / "mmpose")
        self.assertEqual(len(config.fingerprint), 64)

    def test_rejects_batch_size_other_than_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_config(root)
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace(
                    "inference_batch_size: 1", "inference_batch_size: 4"
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CslDailyPoseAnnotationError, "fixed at 1"):
                load_csl_daily_pose_annotation_config(
                    config_path, root, variables={"TEST_ROOT": str(root)}
                )

    def test_rejects_missing_variable_and_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_config(root)
            with self.assertRaisesRegex(CslDailyPoseAnnotationError, "placeholder"):
                load_csl_daily_pose_annotation_config(config_path, root, variables={})
            config_path.write_text(
                config_path.read_text(encoding="utf-8") + "surprise: true\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CslDailyPoseAnnotationError, "unknown keys"):
                load_csl_daily_pose_annotation_config(
                    config_path, root, variables={"TEST_ROOT": str(root)}
                )

    def test_subject_id_parsing(self) -> None:
        self.assertEqual(subject_id_for_sequence("S000000_P0004_T00"), "P0004")
        self.assertIsNone(subject_id_for_sequence("plain_sequence"))

    def test_frame_glob_is_sorted_and_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence_dir = self._write_sequence(
                root,
                "S000001_P0001_T00",
                ["frame_010.jpg", "frame_002.JPG", "frame_001.png", "notes.txt"],
            )
            frames = list_sequence_frames(sequence_dir)

        self.assertEqual(
            [path.name for path in frames],
            ["frame_001.png", "frame_002.JPG", "frame_010.jpg"],
        )

    def test_happy_path_writes_pose_and_manifest_conforming_to_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._load_config(root)
            names = [f"frame_{index:03d}.jpg" for index in range(3)]
            self._write_sequence(root, "S000000_P0004_T00", names)
            self._write_sequence(root, "S000002_P0007_T01", names)
            scripted = {
                "S000000_P0004_T00": _full_score_keypoints(3, z_base=10.0),
                "S000002_P0007_T01": _full_score_keypoints(3, z_base=20.0),
            }
            estimator = _FakeEstimator(scripted)

            result = run_csl_daily_pose_annotation(config, estimator=estimator)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["processed"], 2)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["manifest_rows"], 2)

            pose_path = (
                config.runtime.output_root / "poses" / "S000000_P0004_T00.npy"
            )
            pose = np.load(pose_path)
            self.assertEqual(pose.shape, (3, 2, 24, 3))
            self.assertEqual(pose.dtype, np.float32)
            self.assertTrue(np.isfinite(pose).all())

            manifest_path = config.runtime.output_root / "pose_manifest.jsonl"
            entries = load_pose_manifest(manifest_path)
            self.assertEqual(len(entries), 2)
            first = entries[0]
            self.assertEqual(first.sample_id, "S000000_P0004_T00")
            self.assertEqual(first.sequence_id, "S000000_P0004_T00")
            self.assertEqual(first.subject_id, "P0004")
            self.assertEqual(first.pose_uri, "poses/S000000_P0004_T00.npy")
            # Only loader-contract keys are present.
            for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
                self.assertLessEqual(
                    set(json.loads(raw_line)),
                    {"sample_id", "pose_uri", "pose_sha256", "sequence_id", "subject_id"},
                )

            sidecar = json.loads(
                (config.runtime.output_root / "poses" / "S000000_P0004_T00.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["status"], "completed")
            self.assertEqual(sidecar["qc"]["status"], "passed")
            self.assertEqual(sidecar["source"]["frame_count"], 3)
            self.assertEqual(
                sidecar["model"]["checkpoint_sha256"], "a" * 64
            )
            self.assertEqual(sidecar["artifact"]["sha256"], first.pose_sha256)

            qc_rows = [
                json.loads(line)
                for line in (config.runtime.output_root / "pose_qc.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(qc_rows), 2)
            self.assertEqual(qc_rows[0]["frame_count"], 3)

    def test_score_masking_sets_low_confidence_keypoints_to_nan(self) -> None:
        keypoints, scores = _full_score_keypoints(2)
        scores[:, 9] = 0.4  # left wrist: below the 0.5 threshold
        scores[:, 10] = 0.5  # right wrist: exactly at the threshold, kept
        # Keep depth-center joints 6/7 valid so only the masking shows up.
        reduction = reduce_sequence_poses(
            keypoints, scores, confidence_threshold=0.5
        )

        self.assertTrue(np.isnan(reduction.pose[:, 0, 2]).all())
        self.assertTrue(np.isfinite(reduction.pose[:, 1, 2]).all())
        self.assertIn("left_arm_nan", reduction.qc_reasons)

    def test_v2_reduction_preserves_validity_and_never_emits_nan(self) -> None:
        keypoints, scores = _full_score_keypoints(2)
        scores[:, 91] = 0.1
        keypoints[1, 92, 0] = np.nan
        annotation = build_sequence_annotation_v2(
            keypoints,
            scores,
            confidence_threshold=0.5,
            minimum_valid_joints_per_frame=12,
            minimum_valid_frame_ratio=0.8,
        )

        self.assertTrue(np.isfinite(annotation.canonical_pose).all())
        self.assertFalse(annotation.canonical_valid[:, 0, 3].any())
        np.testing.assert_allclose(
            annotation.canonical_confidence[:, 0, 3], [0.1, 0.1]
        )
        self.assertTrue(annotation.canonical_imputed[:, 0, 3].all())
        self.assertTrue((annotation.canonical_pose[:, 0, 3] == 0.0).all())
        self.assertFalse(annotation.canonical_valid[1, 0, 4])
        self.assertTrue(annotation.canonical_imputed[1, 0, 4])
        self.assertNotEqual(float(annotation.canonical_pose[1, 0, 4, 0]), 0.0)
        self.assertTrue(np.isnan(annotation.native_keypoints_3d[1, 92, 0]))
        self.assertEqual(annotation.frame_mask.dtype, np.bool_)
        self.assertTrue(annotation.frame_mask.all())

    def test_depth_recentering_subtracts_sequence_mean_of_joints_6_and_7(self) -> None:
        keypoints, scores = _full_score_keypoints(3, z_base=10.0)
        original = keypoints.copy()
        expected_center = float(original[:, [6, 7], 2].mean())

        reduction = reduce_sequence_poses(
            keypoints, scores, confidence_threshold=0.5
        )

        self.assertAlmostEqual(reduction.depth_center, expected_center, places=5)
        left_indices = LEFT_JOINT_INDICES
        np.testing.assert_allclose(
            reduction.pose[:, 0, :, 2],
            original[:, left_indices, 2] - np.float32(expected_center),
            rtol=1e-5,
            atol=1e-5,
        )
        # x/y are untouched by the depth centering.
        np.testing.assert_array_equal(
            reduction.pose[:, 1, :, :2], original[:, RIGHT_JOINT_INDICES, :2]
        )
        self.assertEqual(reduction.qc_reasons, ())

    def test_reduction_maps_legacy_59_keypoint_layout(self) -> None:
        keypoints, scores = _full_score_keypoints(1)
        reduction = reduce_sequence_poses(
            keypoints, scores, confidence_threshold=0.5
        )
        # Left side: body joints 5/7/9 then native left-hand joints 91..111.
        np.testing.assert_array_equal(
            reduction.pose[0, 0, :, 0],
            keypoints[0, [5, 7, 9, *range(91, 112)], 0],
        )
        # Right side: body joints 6/8/10 then native right-hand 112..132.
        np.testing.assert_array_equal(
            reduction.pose[0, 1, :, 0],
            keypoints[0, [6, 8, 10, *range(112, 133)], 0],
        )

    def test_nan_arm_sequence_is_skipped_with_structured_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._load_config(root)
            self._write_sequence(root, "S000000_P0004_T00", ["f0.jpg", "f1.jpg"])
            keypoints, scores = _full_score_keypoints(2)
            scores[1, 5] = 0.1  # left shoulder low confidence in one frame
            estimator = _FakeEstimator(
                {"S000000_P0004_T00": (keypoints, scores)}
            )

            result = run_csl_daily_pose_annotation(config, estimator=estimator)

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["skipped_qc"], 1)
            self.assertEqual(result["manifest_rows"], 0)
            self.assertFalse(
                (config.runtime.output_root / "poses" / "S000000_P0004_T00.npy").exists()
            )
            sidecar = json.loads(
                (config.runtime.output_root / "poses" / "S000000_P0004_T00.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["status"], "skipped")
            self.assertEqual(sidecar["qc"]["reasons"], ["left_arm_nan"])
            self.assertIsNone(sidecar["artifact"])

    def test_all_nan_hand_sequence_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._load_config(root)
            self._write_sequence(root, "S000000_P0004_T00", ["f0.jpg"])
            keypoints, scores = _full_score_keypoints(1)
            scores[0, 91:112] = 0.0  # whole left hand below threshold
            estimator = _FakeEstimator(
                {"S000000_P0004_T00": (keypoints, scores)}
            )

            result = run_csl_daily_pose_annotation(config, estimator=estimator)

            self.assertEqual(result["skipped_qc"], 1)
            sidecar = json.loads(
                (config.runtime.output_root / "poses" / "S000000_P0004_T00.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(sidecar["qc"]["reasons"], ["left_hand_all_nan"])

    def test_frame_order_is_deterministic_by_file_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._load_config(root)
            # Creation order intentionally differs from name order.
            self._write_sequence(
                root,
                "S000000_P0004_T00",
                ["frame_002.jpg", "frame_000.jpg", "frame_001.JPG"],
            )
            keypoints, scores = _full_score_keypoints(3)
            estimator = _FakeEstimator(
                {"S000000_P0004_T00": (keypoints, scores)}
            )

            run_csl_daily_pose_annotation(config, estimator=estimator)

            self.assertEqual(
                [path.name for path in estimator.calls],
                ["frame_000.jpg", "frame_001.JPG", "frame_002.jpg"],
            )
            pose = np.load(
                config.runtime.output_root / "poses" / "S000000_P0004_T00.npy"
            )
            # Time axis follows the sorted frame order: x of joint 5 grows by
            # 1000 per scripted frame.
            self.assertTrue(
                np.all(np.diff(pose[:, 0, 0, 0]) > 0)
            )

    def test_restart_skips_finished_sequences_and_never_reprocesses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._load_config(root)
            self._write_sequence(root, "S000000_P0004_T00", ["f0.jpg", "f1.jpg"])
            estimator = _FakeEstimator(
                {"S000000_P0004_T00": _full_score_keypoints(2)}
            )

            first = run_csl_daily_pose_annotation(config, estimator=estimator)
            self.assertEqual(first["processed"], 1)
            pose_bytes = (
                config.runtime.output_root / "poses" / "S000000_P0004_T00.npy"
            ).read_bytes()
            manifest_bytes = (
                config.runtime.output_root / "pose_manifest.jsonl"
            ).read_bytes()

            second = run_csl_daily_pose_annotation(config, estimator=estimator)

            self.assertEqual(second["processed"], 0)
            self.assertEqual(second["skipped_existing"], 1)
            self.assertEqual(second["manifest_rows"], 1)
            self.assertEqual(len(estimator.calls), 2)  # no new estimation calls
            self.assertEqual(
                (config.runtime.output_root / "poses" / "S000000_P0004_T00.npy")
                .read_bytes(),
                pose_bytes,
            )
            self.assertEqual(
                (config.runtime.output_root / "pose_manifest.jsonl").read_bytes(),
                manifest_bytes,
            )

    def test_v2_writes_audit_payload_and_validates_before_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._load_v2_config(root)
            self._write_sequence(root, "S000000_P0004_T00", ["f0.jpg", "f1.jpg"])
            estimator = _FakeEstimator(
                {"S000000_P0004_T00": _full_score_keypoints(2)}
            )

            first = run_csl_daily_pose_annotation(config, estimator=estimator)

            self.assertEqual(first["manifest_rows"], 1)
            pose_path = config.runtime.output_root / "poses" / "S000000_P0004_T00.npy"
            audit_path = config.runtime.output_root / "samples" / "S000000_P0004_T00.npz"
            self.assertTrue(validate_annotation_v2_audit(audit_path, pose_path=pose_path))
            with np.load(audit_path, allow_pickle=False) as arrays:
                self.assertEqual(
                    set(arrays.files),
                    {
                        "native_keypoints_3d",
                        "native_keypoint_scores",
                        "canonical_pose",
                        "canonical_confidence",
                        "canonical_valid",
                        "canonical_imputed",
                        "frame_mask",
                    },
                )
                self.assertTrue(np.isfinite(arrays["canonical_pose"]).all())

            second = run_csl_daily_pose_annotation(config, estimator=estimator)
            self.assertEqual(second["skipped_existing"], 1)
            self.assertEqual(len(estimator.calls), 2)

            with np.load(audit_path, allow_pickle=False) as arrays:
                tampered = {name: arrays[name] for name in arrays.files}
            tampered["canonical_pose"] = tampered["canonical_pose"].copy()
            tampered["canonical_pose"][0, 0, 0, 0] += 1.0
            np.savez_compressed(audit_path, **tampered)
            with self.assertRaisesRegex(
                CslDailyPoseAnnotationConflictError, "refusing to overwrite"
            ):
                run_csl_daily_pose_annotation(config, estimator=estimator)

    def test_orphan_artifact_is_a_conflict_never_clobbered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._load_config(root)
            self._write_sequence(root, "S000000_P0004_T00", ["f0.jpg"])
            estimator = _FakeEstimator(
                {"S000000_P0004_T00": _full_score_keypoints(1)}
            )
            run_csl_daily_pose_annotation(config, estimator=estimator)
            # Corrupt the artifact: the sidecar identity no longer matches.
            npy_path = config.runtime.output_root / "poses" / "S000000_P0004_T00.npy"
            npy_path.write_bytes(b"corrupted")

            with self.assertRaisesRegex(
                CslDailyPoseAnnotationConflictError, "refusing to overwrite"
            ):
                run_csl_daily_pose_annotation(config, estimator=estimator)
            self.assertEqual(npy_path.read_bytes(), b"corrupted")

    def test_estimator_failure_is_recorded_and_run_continues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._load_config(root)
            self._write_sequence(root, "S000000_P0004_T00", ["f0.jpg"])
            self._write_sequence(root, "S000001_P0004_T00", ["f0.jpg"])

            class _FlakyEstimator(_FakeEstimator):
                def estimate_frame(self, image_path: Path) -> FramePosePrediction:
                    if image_path.parent.name == "S000000_P0004_T00":
                        raise RuntimeError("decoder exploded")
                    return super().estimate_frame(image_path)

            estimator = _FlakyEstimator(
                {"S000001_P0004_T00": _full_score_keypoints(1)}
            )

            result = run_csl_daily_pose_annotation(config, estimator=estimator)

            self.assertEqual(result["status"], "completed_with_failures")
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["processed"], 1)
            self.assertEqual(result["manifest_rows"], 1)
            run_record = json.loads(
                Path(result["run_record"]).read_text(encoding="utf-8")
            )
            failed = [
                entry
                for entry in run_record["sequences"]
                if entry["outcome"] == "failed"
            ]
            self.assertEqual(failed[0]["sequence_id"], "S000000_P0004_T00")
            self.assertIn("decoder exploded", failed[0]["error"])

    def test_sequence_selection_and_max_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._load_config(root)
            for index in range(3):
                self._write_sequence(root, f"S00000{index}_P0004_T00", ["f0.jpg"])
            estimator = _FakeEstimator(
                {
                    f"S00000{index}_P0004_T00": _full_score_keypoints(1)
                    for index in range(3)
                }
            )

            limited = run_csl_daily_pose_annotation(
                config, estimator=estimator, max_sequences=2
            )
            self.assertEqual(limited["status"], "limit_reached")
            self.assertEqual(limited["processed"], 2)

            selected = run_csl_daily_pose_annotation(
                config, estimator=estimator, sequence_ids=["S000002_P0004_T00"]
            )
            self.assertEqual(selected["processed"], 1)
            self.assertEqual(selected["manifest_rows"], 3)

            with self.assertRaisesRegex(
                CslDailyPoseAnnotationError, "not found"
            ):
                run_csl_daily_pose_annotation(
                    config, estimator=estimator, sequence_ids=["S999999_P0004_T00"]
                )

    def test_sequence_ids_must_match_manifest_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence_root = root / "sequences"
            (sequence_root / "bad id").mkdir(parents=True)
            with self.assertRaisesRegex(CslDailyPoseAnnotationError, "must match"):
                discover_sequences(sequence_root)


if __name__ == "__main__":
    unittest.main()
