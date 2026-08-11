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
    passed_csl_news_integrity_archives,
    scan_csl_news_source_integrity,
)


class CslNewsIntegrityTest(unittest.TestCase):
    def _write_config(self, root: Path) -> Path:
        config = root / "integrity.yaml"
        config.write_text(
            """schema_version: mmprism.csl_news_source_integrity_config.v1
source:
  data_root: ${MMPRISM_DATA_ROOT}
  archive_root: incoming/archives
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
