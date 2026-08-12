from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from mmprism.data import load_csl_news_annotation_config
from mmprism.data.csl_news_scheduler import (
    CslNewsSchedulerError,
    build_csl_news_scheduler_status,
    claim_csl_news_annotation_archive,
    initialize_csl_news_scheduler,
    release_csl_news_annotation_lease,
    renew_csl_news_annotation_lease,
    scheduler_root,
    set_csl_news_scheduler_state,
)


class CslNewsSchedulerTest(unittest.TestCase):
    def _config(self, root: Path):
        path = root / "annotation.yaml"
        path.write_text(
            f"""schema_version: mmprism.csl_news_pose_annotation.v1
source:
  archive_root: {root / 'archives'}
  labels_path: {root / 'labels.json'}
  source_id: fixture
  source_revision: revision
  expected_archive_count: 2
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
  poll_seconds: 1
  min_free_bytes: 0
  max_consecutive_oom: 2
""",
            encoding="utf-8",
        )
        return load_csl_news_annotation_config(path, root)

    def _registry(self, root: Path, config) -> Path:
        config.source.archive_root.mkdir()
        config.source.labels_path.write_text("[]", encoding="utf-8")
        archives: dict[str, dict[str, object]] = {}
        for archive_id, member_count in ((1, 1), (2, 2)):
            archive_path = config.source.archive_root / f"archive_{archive_id:03d}.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index in range(member_count):
                    archive.writestr(f"sample_{archive_id}_{index}.mp4", b"video")
            stat = archive_path.stat()
            key = f"{archive_id:03d}"
            archives[key] = {
                "archive_id": key,
                "archive_name": archive_path.name,
                "archive_path_relative": archive_path.name,
                "source_kind": "primary",
                "status": "passed",
                "source_present": True,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": f"{archive_id:x}" * 64,
                "video_count": member_count,
            }
        registry = root / "registry.json"
        registry.write_text(
            json.dumps(
                {
                    "schema_version": "mmprism.csl_news_source_integrity_registry.v2",
                    "source": {
                        "source_id": "fixture",
                        "source_revision": "revision",
                        "labels_sha256": "c" * 64,
                    },
                    "archives": archives,
                }
            ),
            encoding="utf-8",
        )
        return registry

    def test_initialize_pauses_and_resume_claims_one_archive_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            registry = self._registry(root, config)
            initialized = initialize_csl_news_scheduler(config, lease_seconds=60)
            paused_lease, paused = claim_csl_news_annotation_archive(
                config, integrity_registry_path=registry, worker_id="worker-a"
            )
            resumed = set_csl_news_scheduler_state(config, state="running", reason="test")
            first, claimed = claim_csl_news_annotation_archive(
                config, integrity_registry_path=registry, worker_id="worker-a"
            )
            assert first is not None
            second, _ = claim_csl_news_annotation_archive(
                config, integrity_registry_path=registry, worker_id="worker-b"
            )
            assert second is not None

        self.assertEqual(initialized["control"]["state"], "paused")
        self.assertEqual(paused["state"], "paused")
        self.assertIsNone(paused_lease)
        self.assertEqual(resumed["status"], "running")
        self.assertEqual(claimed["state"], "claimed")
        self.assertEqual(first.archive_id, 2)
        self.assertEqual(second.archive_id, 1)

    def test_pause_prevents_next_sample_but_renews_existing_lease(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            registry = self._registry(root, config)
            initialize_csl_news_scheduler(config, lease_seconds=60)
            set_csl_news_scheduler_state(config, state="running")
            lease, _ = claim_csl_news_annotation_archive(
                config, integrity_registry_path=registry, worker_id="worker-a"
            )
            assert lease is not None
            set_csl_news_scheduler_state(config, state="paused")
            should_continue = renew_csl_news_annotation_lease(config, lease)
            history = scheduler_root(config) / "leases" / "history" / "test.json"
            release_csl_news_annotation_lease(config, lease, result={"status": "paused"})
            history_created = history.parent.is_dir()

        self.assertFalse(should_continue)
        self.assertFalse(lease.lease_path.exists())
        self.assertTrue(history_created)

    def test_stale_lease_is_preserved_then_reclaimed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            registry = self._registry(root, config)
            initialize_csl_news_scheduler(config, lease_seconds=60)
            set_csl_news_scheduler_state(config, state="running")
            lease, _ = claim_csl_news_annotation_archive(
                config, integrity_registry_path=registry, worker_id="dead-worker"
            )
            assert lease is not None
            payload = json.loads(lease.lease_path.read_text(encoding="utf-8"))
            payload["heartbeat_unix_seconds"] = 0
            lease.lease_path.write_text(json.dumps(payload), encoding="utf-8")
            with patch("mmprism.data.csl_news_scheduler.time.time", return_value=1000.0):
                recovered, claim = claim_csl_news_annotation_archive(
                    config, integrity_registry_path=registry, worker_id="replacement"
                )
            assert recovered is not None
            expired = list((scheduler_root(config) / "leases" / "expired").glob("*.json"))

        self.assertEqual(claim["stale_leases_recovered"], 1)
        self.assertEqual(recovered.archive_id, lease.archive_id)
        self.assertEqual(len(expired), 1)

    def test_status_and_identity_mismatch_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            registry = self._registry(root, config)
            initialize_csl_news_scheduler(config, lease_seconds=60)
            status = build_csl_news_scheduler_status(
                config, integrity_registry_path=registry
            )
            control = scheduler_root(config) / "control.json"
            payload = json.loads(control.read_text(encoding="utf-8"))
            payload["identity"]["source_id"] = "other"
            control.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(CslNewsSchedulerError):
                set_csl_news_scheduler_state(config, state="running")

        self.assertEqual(status["control"]["state"], "paused")
        self.assertEqual(status["queue"]["eligible_archive_count"], 2)


if __name__ == "__main__":
    unittest.main()
