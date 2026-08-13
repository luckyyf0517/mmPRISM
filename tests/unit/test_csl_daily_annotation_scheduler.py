from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mmprism.data import load_csl_daily_pose_annotation_config
from mmprism.data.csl_daily_annotation_scheduler import (
    CslDailySchedulerError,
    build_csl_daily_scheduler_status,
    claim_csl_daily_annotation_sequence,
    finalize_csl_daily_annotation_v2,
    initialize_csl_daily_scheduler,
    release_csl_daily_annotation_lease,
    run_csl_daily_annotation_scheduled_worker,
    scheduler_root,
    set_csl_daily_scheduler_state,
)
from mmprism.data.csl_daily_pose_annotation import (
    FramePosePrediction,
    run_csl_daily_pose_annotation,
)


class _FakeEstimator:
    def estimate_frame(self, _: Path) -> FramePosePrediction:
        joints = np.arange(133, dtype=np.float32)
        keypoints = np.stack((joints, joints + 1.0, joints / 100.0), axis=1)
        return FramePosePrediction(keypoints=keypoints, scores=np.ones(133, dtype=np.float32))

    def runtime_metadata(self) -> dict[str, object]:
        return {"device": "fake"}


class CslDailyAnnotationSchedulerTest(unittest.TestCase):
    def _config(self, root: Path):
        config_path = root / "annotation.yaml"
        config_path.write_text(
            f"""schema_version: mmprism.csl_daily_pose_annotation.v2
source:
  sequence_root: {root / 'sequences'}
  source_id: fixture
model:
  mmpose_root: {root / 'mmpose'}
  mmpose_commit: {'c' * 40}
  project_dir: {root / 'project'}
  config_path: {root / 'model.py'}
  checkpoint_path: {root / 'model.pth'}
  checkpoint_sha256: {'a' * 64}
  device: cuda:0
transform:
  confidence_threshold: 0.5
  minimum_valid_joints_per_frame: 12
  minimum_valid_frame_ratio: 0.8
runtime:
  output_root: {root / 'output'}
  inference_batch_size: 1
""",
            encoding="utf-8",
        )
        return load_csl_daily_pose_annotation_config(config_path, root, variables={})

    def _sequence(self, root: Path, sequence_id: str) -> None:
        path = root / "sequences" / sequence_id
        path.mkdir(parents=True)
        (path / "frame.jpg").write_bytes(b"frame")

    def test_paused_queue_claims_no_sequence_then_leases_are_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._sequence(root, "S000000_P0004_T00")
            self._sequence(root, "S000001_P0004_T00")
            config = self._config(root)
            initialized = initialize_csl_daily_scheduler(config, lease_seconds=60)
            paused_lease, paused = claim_csl_daily_annotation_sequence(config, worker_id="a")
            resumed = set_csl_daily_scheduler_state(config, state="running")
            first, _ = claim_csl_daily_annotation_sequence(config, worker_id="a")
            second, _ = claim_csl_daily_annotation_sequence(config, worker_id="b")
            assert first is not None and second is not None

        self.assertEqual(initialized["control"]["state"], "paused")
        self.assertIsNone(paused_lease)
        self.assertEqual(paused["state"], "paused")
        self.assertEqual(resumed["status"], "running")
        self.assertNotEqual(first.sequence_id, second.sequence_id)

    def test_stale_lease_is_retained_then_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._sequence(root, "S000000_P0004_T00")
            config = self._config(root)
            initialize_csl_daily_scheduler(config, lease_seconds=60)
            set_csl_daily_scheduler_state(config, state="running")
            lease, _ = claim_csl_daily_annotation_sequence(config, worker_id="dead")
            assert lease is not None
            payload = json.loads(lease.lease_path.read_text(encoding="utf-8"))
            payload["heartbeat_unix_seconds"] = 0
            lease.lease_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "mmprism.data.csl_daily_annotation_scheduler.time.time", return_value=1000.0
            ):
                replacement, report = claim_csl_daily_annotation_sequence(config, worker_id="next")
            assert replacement is not None
            expired = list((scheduler_root(config) / "leases" / "expired").glob("*.json"))

        self.assertEqual(report["stale_leases_recovered"], 1)
        self.assertEqual(replacement.sequence_id, lease.sequence_id)
        self.assertEqual(len(expired), 1)

    def test_finalize_requires_paused_queue_without_leases_and_writes_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._sequence(root, "S000000_P0004_T00")
            config = self._config(root)
            initialize_csl_daily_scheduler(config, lease_seconds=60)
            set_csl_daily_scheduler_state(config, state="running")
            lease, _ = claim_csl_daily_annotation_sequence(config, worker_id="worker")
            assert lease is not None
            with self.assertRaisesRegex(CslDailySchedulerError, "paused scheduler"):
                finalize_csl_daily_annotation_v2(config)
            result = run_csl_daily_pose_annotation(
                config,
                estimator=_FakeEstimator(),
                sequence_ids=[lease.sequence_id],
                rewrite_manifests=False,
            )
            release_csl_daily_annotation_lease(config, lease, result=result)
            set_csl_daily_scheduler_state(config, state="paused")
            finalized = finalize_csl_daily_annotation_v2(config)
            coverage = json.loads((config.runtime.output_root / "coverage.json").read_text())

        self.assertEqual(finalized["status"], "finalized")
        self.assertEqual(finalized["completed_eligible"], 1)
        self.assertEqual(coverage["coverage"]["unfinished_sequences"], 0)

    def test_status_detects_identity_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._sequence(root, "S000000_P0004_T00")
            config = self._config(root)
            initialize_csl_daily_scheduler(config, lease_seconds=60)
            status = build_csl_daily_scheduler_status(config)
            control = scheduler_root(config) / "control.json"
            payload = json.loads(control.read_text(encoding="utf-8"))
            payload["identity"]["source_id"] = "tampered"
            control.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(CslDailySchedulerError):
                set_csl_daily_scheduler_state(config, state="running")

        self.assertEqual(status["coverage"]["source_sequences"], 1)

    def test_worker_quarantines_per_sequence_failure_instead_of_retrying(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._sequence(root, "S000000_P0004_T00")
            config = self._config(root)
            initialize_csl_daily_scheduler(config, lease_seconds=60)
            set_csl_daily_scheduler_state(config, state="running")
            class _BadEstimator:
                def estimate_frame(self, _: Path) -> FramePosePrediction:
                    raise RuntimeError("bad frame")

                def runtime_metadata(self) -> dict[str, object]:
                    return {"device": "bad"}

            with patch(
                "mmprism.data.csl_daily_annotation_scheduler.MMPoseRtmw3dFrameEstimator",
                return_value=_BadEstimator(),
            ):
                result = run_csl_daily_annotation_scheduled_worker(config, once=True)
            quarantine = config.runtime.output_root / "quarantine" / "S000000_P0004_T00.json"
            quarantine_exists = quarantine.is_file()

        self.assertEqual(result["sequences"], 1)
        self.assertTrue(quarantine_exists)

    def test_worker_initialization_error_releases_lease_without_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._sequence(root, "S000000_P0004_T00")
            config = self._config(root)
            initialize_csl_daily_scheduler(config, lease_seconds=60)
            set_csl_daily_scheduler_state(config, state="running")
            with (
                patch(
                    "mmprism.data.csl_daily_annotation_scheduler.MMPoseRtmw3dFrameEstimator",
                    side_effect=RuntimeError("model construction failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "model construction failed"),
            ):
                run_csl_daily_annotation_scheduled_worker(config, once=True)
            history = list((scheduler_root(config) / "leases" / "history").glob("*.json"))
            quarantine = list((config.runtime.output_root / "quarantine").glob("*.json"))

        self.assertEqual(len(history), 1)
        self.assertEqual(len(quarantine), 0)


if __name__ == "__main__":
    unittest.main()
