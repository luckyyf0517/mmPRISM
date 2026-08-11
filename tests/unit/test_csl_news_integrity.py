import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from mmprism.data import (
    CslNewsIntegrityError,
    load_csl_news_integrity_config,
    load_csl_news_integrity_registry,
    load_csl_news_integrity_registry_snapshot,
    passed_csl_news_integrity_archives,
    scan_csl_news_source_integrity,
)


class CslNewsIntegrityTest(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config = root / "integrity.yaml"
        config.write_text(
            """schema_version: mmprism.csl_news_source_integrity_config.v2
source:
  data_root: ${MMPRISM_DATA_ROOT}
  archive_root: incoming/archives
  replacement_archives:
    "002": replacements/recovery_v1/rgb_archives/archive_002.zip
  labels_path: incoming/labels.json
  source_id: fixture
  source_revision: revision
  expected_archive_count: 3
validation:
  verify_crc: true
  decode_sample_count: 0
output:
  registry_path: manifests/integrity/registry.json
  audit_root: manifests/integrity/audits
  scratch_root: cache/integrity
""",
            encoding="utf-8",
        )
        return config

    def _prepare_sources(self, root: Path) -> None:
        archive_root = root / "incoming" / "archives"
        archive_root.mkdir(parents=True)
        with zipfile.ZipFile(archive_root / "archive_001.zip", "w") as archive:
            archive.writestr("first.mp4", b"first-video")
            archive.writestr("second.mp4", b"second-video")
        (archive_root / "archive_002.zip").write_bytes(
            b"PK incomplete archive without central directory"
        )
        (root / "incoming" / "labels.json").write_text(
            json.dumps(
                [
                    {"video": "first.mp4", "text": "第一条"},
                    {"video": "second.mp4", "text": "第二条"},
                    {"video": "third.mp4", "text": "第三条"},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _runtime(self, *, dirty: bool = False) -> dict[str, object]:
        return {"git": {"commit": "a" * 40, "dirty": dirty}}

    def test_builds_and_reuses_cumulative_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare_sources(root)
            with mock.patch.dict("os.environ", {"MMPRISM_DATA_ROOT": str(root)}):
                config = load_csl_news_integrity_config(self._write_config(root))
            registry = scan_csl_news_source_integrity(
                config, runtime_report=self._runtime()
            )

            self.assertEqual(registry["status"], "partial_with_failures")
            self.assertEqual(registry["summary"]["passed_archive_ids"], ["001"])
            self.assertEqual(registry["summary"]["failed_archive_ids"], ["002"])
            self.assertEqual(registry["summary"]["passed_video_count"], 2)
            self.assertEqual(registry["last_scan"]["audited_count"], 2)
            serialized = json.dumps(registry)
            self.assertNotIn(str(root), serialized)

            loaded = load_csl_news_integrity_registry(
                config.registry_path,
                source_id="fixture",
                source_revision="revision",
            )
            passed = passed_csl_news_integrity_archives(loaded)
            self.assertEqual(list(passed), [1])
            self.assertEqual(passed[1].video_count, 2)
            self.assertEqual(passed[1].archive_path_relative, Path("archive_001.zip"))
            self.assertEqual(passed[1].source_kind, "primary")
            snapshot, snapshot_sha256 = load_csl_news_integrity_registry_snapshot(
                config.registry_path,
                source_id="fixture",
                source_revision="revision",
            )
            self.assertEqual(snapshot, loaded)
            self.assertEqual(
                snapshot_sha256,
                hashlib.sha256(config.registry_path.read_bytes()).hexdigest(),
            )
            for entry in loaded["archives"].values():
                audit_path = root / entry["audit"]["path"]
                self.assertTrue(audit_path.is_file())

            repeated = scan_csl_news_source_integrity(
                config, runtime_report=self._runtime()
            )
            self.assertEqual(repeated["last_scan"]["audited_count"], 0)
            self.assertEqual(repeated["last_scan"]["reused_count"], 2)

    def test_incrementally_audits_new_final_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare_sources(root)
            with mock.patch.dict("os.environ", {"MMPRISM_DATA_ROOT": str(root)}):
                config = load_csl_news_integrity_config(self._write_config(root))
            scan_csl_news_source_integrity(config, runtime_report=self._runtime())
            with zipfile.ZipFile(
                config.archive_root / "archive_003.zip", "w"
            ) as archive:
                archive.writestr("third.mp4", b"third-video")

            registry = scan_csl_news_source_integrity(
                config,
                runtime_report=self._runtime(),
                max_new_archives=1,
            )

        self.assertEqual(registry["last_scan"]["audited_count"], 1)
        self.assertEqual(registry["summary"]["passed_archive_ids"], ["001", "003"])
        self.assertEqual(registry["summary"]["passed_video_count"], 3)

    def test_replacement_supersedes_failed_primary_without_modifying_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare_sources(root)
            with mock.patch.dict("os.environ", {"MMPRISM_DATA_ROOT": str(root)}):
                config = load_csl_news_integrity_config(self._write_config(root))
            primary = config.archive_root / "archive_002.zip"
            primary_identity = (
                primary.stat().st_size,
                primary.stat().st_mtime_ns,
                hashlib.sha256(primary.read_bytes()).hexdigest(),
            )
            first = scan_csl_news_source_integrity(
                config, runtime_report=self._runtime()
            )
            self.assertEqual(first["archives"]["002"]["status"], "failed")

            replacement_relative = config.replacement_archives_relative[2]
            replacement = config.archive_root / replacement_relative
            replacement.parent.mkdir(parents=True)
            with zipfile.ZipFile(replacement, "w") as archive:
                archive.writestr("third.mp4", b"replacement-video")

            recovered = scan_csl_news_source_integrity(
                config, runtime_report=self._runtime()
            )

            self.assertEqual(recovered["archives"]["002"]["status"], "passed")
            self.assertEqual(
                recovered["archives"]["002"]["archive_path_relative"],
                replacement_relative.as_posix(),
            )
            self.assertEqual(
                recovered["summary"]["selected_replacement_archive_ids"], ["002"]
            )
            self.assertEqual(recovered["last_scan"]["audited_count"], 1)
            passed = passed_csl_news_integrity_archives(recovered)
            self.assertEqual(passed[2].archive_path_relative, replacement_relative)
            self.assertEqual(passed[2].source_kind, "replacement")
            self.assertEqual(
                (
                    primary.stat().st_size,
                    primary.stat().st_mtime_ns,
                    hashlib.sha256(primary.read_bytes()).hexdigest(),
                ),
                primary_identity,
            )

    def test_rejects_non_nested_replacement_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._write_config(root)
            text = config_path.read_text(encoding="utf-8").replace(
                "replacements/recovery_v1/rgb_archives/archive_002.zip",
                "archive_002.zip",
            )
            config_path.write_text(text, encoding="utf-8")
            with (
                mock.patch.dict("os.environ", {"MMPRISM_DATA_ROOT": str(root)}),
                self.assertRaisesRegex(CslNewsIntegrityError, "nested path"),
            ):
                load_csl_news_integrity_config(config_path)

    def test_requires_clean_git_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare_sources(root)
            with mock.patch.dict("os.environ", {"MMPRISM_DATA_ROOT": str(root)}):
                config = load_csl_news_integrity_config(self._write_config(root))

            with self.assertRaisesRegex(CslNewsIntegrityError, "clean Git"):
                scan_csl_news_source_integrity(
                    config, runtime_report=self._runtime(dirty=True)
                )

    def test_reaudits_existing_archives_when_labels_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._prepare_sources(root)
            with mock.patch.dict("os.environ", {"MMPRISM_DATA_ROOT": str(root)}):
                config = load_csl_news_integrity_config(self._write_config(root))
            scan_csl_news_source_integrity(config, runtime_report=self._runtime())
            labels = json.loads(config.labels_path.read_text(encoding="utf-8"))
            labels[0]["text"] = "更新后的第一条"
            config.labels_path.write_text(
                json.dumps(labels, ensure_ascii=False), encoding="utf-8"
            )

            registry = scan_csl_news_source_integrity(
                config, runtime_report=self._runtime()
            )

        self.assertEqual(registry["last_scan"]["audited_count"], 2)
        self.assertEqual(registry["last_scan"]["reused_count"], 0)


if __name__ == "__main__":
    unittest.main()
