import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from mmprism.contracts import validate_manifest
from mmprism.data import (
    CslNewsSourceManifestConfig,
    CslNewsSourceManifestError,
    build_csl_news_source_manifest_snapshot,
    load_csl_news_source_manifest_config,
)


class CslNewsSourceManifestTest(unittest.TestCase):
    def _config(self, root: Path) -> CslNewsSourceManifestConfig:
        config_path = root / "manifest.yaml"
        config_path.write_text(
            """schema_version: mmprism.csl_news_source_manifest.v1
source:
  data_root: ${MMPRISM_TEST_DATA_ROOT}
  archive_root: incoming/archives
  labels_path: incoming/metadata/labels.json
  source_id: fixture:csl-news
  source_revision: revision
  expected_archive_count: 2
validation:
  verify_crc: true
  minimum_free_bytes: 0
output:
  snapshot_root: manifests/csl_news/source_manifest_v1
""",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"MMPRISM_TEST_DATA_ROOT": str(root)}):
            return load_csl_news_source_manifest_config(config_path)

    def _source(self, root: Path) -> None:
        archive_root = root / "incoming" / "archives"
        archive_root.mkdir(parents=True)
        video_names = [
            "Common-Concerns_20200101_0-100_1.mp4",
            "20200102_Dragon-TV__0-100_2.mp4",
        ]
        with zipfile.ZipFile(archive_root / "archive_001.zip", "w") as archive:
            for video_name in video_names:
                archive.writestr(video_name, b"encoded-video-fixture")
        metadata_root = root / "incoming" / "metadata"
        metadata_root.mkdir(parents=True)
        metadata_root.joinpath("labels.json").write_text(
            json.dumps(
                [
                    {
                        "video": video_name,
                        "pose": video_name.replace(".mp4", ".pkl"),
                        "text": f"文本{index}",
                    }
                    for index, video_name in enumerate(video_names, start=1)
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def _runtime(self, *, dirty: bool = False) -> dict[str, object]:
        return {"git": {"commit": "a" * 40, "dirty": dirty}, "python": "3.12"}

    def test_builds_atomic_portable_partial_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root)
            config = self._config(root)
            receipt = build_csl_news_source_manifest_snapshot(
                config,
                runtime_report=self._runtime(),
                snapshot_id="fixture",
            )
            manifest_path = Path(receipt["manifest_path"])
            summary = validate_manifest(manifest_path)
            records = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
            ]
            manifest_text = manifest_path.read_text(encoding="utf-8")
            snapshot_summary = json.loads(
                Path(receipt["summary_path"]).read_text(encoding="utf-8")
            )

        self.assertEqual(receipt["status"], "partial")
        self.assertEqual(receipt["record_count"], 2)
        self.assertEqual(summary.modalities, ("caption", "video"))
        self.assertTrue(records[0]["modalities"]["video"]["uri"].startswith("zip://"))
        captions_by_sequence = {
            record["sequence_id"]: record["modalities"]["caption"]["text"]
            for record in records
        }
        self.assertEqual(
            captions_by_sequence["Common-Concerns_20200101_0-100_1"], "文本1"
        )
        self.assertEqual(captions_by_sequence["20200102_Dragon-TV__0-100_2"], "文本2")
        self.assertNotIn(str(root), manifest_text)
        self.assertEqual(snapshot_summary["source"]["unrepresented_label_count"], 0)
        self.assertEqual(snapshot_summary["runtime"]["git"]["commit"], "a" * 40)
        self.assertEqual(
            len(records[0]["provenance"]["labels_sha256"]),
            64,
        )

    def test_rejects_archive_video_without_canonical_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root)
            labels_path = root / "incoming" / "metadata" / "labels.json"
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            labels_path.write_text(json.dumps(labels[:1]), encoding="utf-8")
            config = self._config(root)

            with self.assertRaisesRegex(
                CslNewsSourceManifestError, "videos have no JSON label"
            ):
                build_csl_news_source_manifest_snapshot(
                    config,
                    runtime_report=self._runtime(),
                    snapshot_id="missing-label",
                )

    def test_requires_clean_git_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root)
            config = self._config(root)
            with self.assertRaisesRegex(CslNewsSourceManifestError, "clean Git"):
                build_csl_news_source_manifest_snapshot(
                    config,
                    runtime_report=self._runtime(dirty=True),
                    snapshot_id="dirty",
                )

    def test_rejects_absolute_storage_relationships(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "manifest.yaml"
            config_path.write_text(
                f"""schema_version: mmprism.csl_news_source_manifest.v1
source:
  data_root: {root}
  archive_root: /absolute/archives
  labels_path: incoming/labels.json
  source_id: fixture
  source_revision: revision
  expected_archive_count: 1
validation:
  verify_crc: false
  minimum_free_bytes: 0
output:
  snapshot_root: manifests
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                CslNewsSourceManifestError, "must be relative"
            ):
                load_csl_news_source_manifest_config(config_path)


if __name__ == "__main__":
    unittest.main()
