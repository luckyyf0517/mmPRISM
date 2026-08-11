import hashlib
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
            """schema_version: mmprism.csl_news_source_manifest.v2
source:
  data_root: ${MMPRISM_TEST_DATA_ROOT}
  archive_root: incoming/archives
  labels_path: incoming/metadata/labels.json
  integrity_registry: manifests/integrity/registry.json
  source_id: fixture:csl-news
  source_revision: revision
  expected_archive_count: 2
validation:
  verify_crc: true
  minimum_free_bytes: 0
output:
  snapshot_root: manifests/csl_news/source_manifest_v2
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
        self._write_registry(root)

    def _write_registry(
        self,
        root: Path,
        *,
        archive_relative: str = "archive_001.zip",
        source_kind: str = "primary",
    ) -> Path:
        archive_root = root / "incoming" / "archives"
        archive_path = archive_root / archive_relative
        labels_path = root / "incoming" / "metadata" / "labels.json"
        archive_stat = archive_path.stat()
        archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        labels_sha256 = hashlib.sha256(labels_path.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive_path, "r") as archive:
            video_count = sum(
                not member.is_dir() and member.filename.lower().endswith(".mp4")
                for member in archive.infolist()
            )
        registry = {
            "schema_version": "mmprism.csl_news_source_integrity_registry.v2",
            "source": {
                "archive_root": "incoming/archives",
                "expected_archive_count": 2,
                "labels_sha256": labels_sha256,
                "source_id": "fixture:csl-news",
                "source_revision": "revision",
            },
            "archives": {
                "001": {
                    "archive_id": "001",
                    "archive_name": "archive_001.zip",
                    "archive_path_relative": archive_relative,
                    "source_kind": source_kind,
                    "source_present": True,
                    "status": "passed",
                    "size_bytes": archive_stat.st_size,
                    "mtime_ns": archive_stat.st_mtime_ns,
                    "sha256": archive_sha256,
                    "video_count": video_count,
                    "audit": {
                        "path": "manifests/integrity/audits/archive_001.json",
                        "sha256": "b" * 64,
                    },
                    "builder_commit": "c" * 40,
                    "audited_at": "2026-08-11T00:00:00+00:00",
                }
            },
            "summary": {
                "passed_archive_ids": ["001"],
                "passed_count": 1,
                "passed_video_count": video_count,
                "failed_archive_ids": [],
                "failed_count": 0,
                "present_final_count": 1,
            },
        }
        registry_path = root / "manifests" / "integrity" / "registry.json"
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(
            json.dumps(registry, sort_keys=True) + "\n", encoding="utf-8"
        )
        return registry_path

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
            snapshot_root = Path(receipt["snapshot_dir"])
            registry_source = root / "manifests" / "integrity" / "registry.json"
            registry_copy = snapshot_root / "integrity_registry.json"
            checksums = {
                name: digest
                for digest, name in (
                    line.split("  ", maxsplit=1)
                    for line in (snapshot_root / "SHA256SUMS")
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
            }
            registry_bytes_match = (
                registry_copy.read_bytes() == registry_source.read_bytes()
            )
            checksum_results = {
                name: hashlib.sha256((snapshot_root / name).read_bytes()).hexdigest()
                for name in checksums
            }

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
        self.assertEqual(records[0]["provenance"]["archive_source_kind"], "primary")
        self.assertTrue(registry_bytes_match)
        self.assertEqual(
            receipt["integrity_registry_sha256"],
            checksums["integrity_registry.json"],
        )
        for name, digest in checksums.items():
            self.assertEqual(checksum_results[name], digest)

    def test_uses_exact_replacement_path_in_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root)
            archive_root = root / "incoming" / "archives"
            primary = archive_root / "archive_001.zip"
            replacement = (
                archive_root / "replacements" / "recovery" / "archive_001.zip"
            )
            replacement.parent.mkdir(parents=True)
            replacement.write_bytes(primary.read_bytes())
            primary.write_bytes(b"preserved-corrupt-primary")
            self._write_registry(
                root,
                archive_relative="replacements/recovery/archive_001.zip",
                source_kind="replacement",
            )
            receipt = build_csl_news_source_manifest_snapshot(
                self._config(root),
                runtime_report=self._runtime(),
                snapshot_id="replacement",
            )
            records = [
                json.loads(line)
                for line in Path(receipt["manifest_path"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(receipt["record_count"], 2)
        self.assertTrue(
            records[0]["modalities"]["video"]["uri"].startswith(
                "zip://replacements/recovery/archive_001.zip!/"
            )
        )
        provenance = records[0]["provenance"]
        self.assertEqual(provenance["archive_source_kind"], "replacement")
        self.assertEqual(
            provenance["archive_path_relative"],
            "replacements/recovery/archive_001.zip",
        )

    def test_rejects_registered_archive_stat_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root)
            archive_path = root / "incoming" / "archives" / "archive_001.zip"
            with archive_path.open("ab") as stream:
                stream.write(b"drift")
            with self.assertRaisesRegex(
                CslNewsSourceManifestError, "stat identity differs"
            ):
                build_csl_news_source_manifest_snapshot(
                    self._config(root),
                    runtime_report=self._runtime(),
                    snapshot_id="drift",
                )

    def test_rejects_registered_archive_content_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root)
            archive_path = root / "incoming" / "archives" / "archive_001.zip"
            archive_stat = archive_path.stat()
            content = bytearray(archive_path.read_bytes())
            content[-1] ^= 0x01
            archive_path.write_bytes(content)
            os.utime(
                archive_path,
                ns=(archive_stat.st_atime_ns, archive_stat.st_mtime_ns),
            )
            with self.assertRaisesRegex(
                CslNewsSourceManifestError, "SHA-256 differs"
            ):
                build_csl_news_source_manifest_snapshot(
                    self._config(root),
                    runtime_report=self._runtime(),
                    snapshot_id="content-drift",
                )

    def test_rejects_registered_archive_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root)
            archive_root = root / "incoming" / "archives"
            replacement = (
                archive_root / "replacements" / "recovery" / "archive_001.zip"
            )
            replacement.parent.mkdir(parents=True)
            replacement.symlink_to(archive_root / "archive_001.zip")
            self._write_registry(
                root,
                archive_relative="replacements/recovery/archive_001.zip",
                source_kind="replacement",
            )
            with self.assertRaisesRegex(
                CslNewsSourceManifestError, "not a regular file"
            ):
                build_csl_news_source_manifest_snapshot(
                    self._config(root),
                    runtime_report=self._runtime(),
                    snapshot_id="symlink",
                )

    def test_rejects_archive_video_without_canonical_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._source(root)
            labels_path = root / "incoming" / "metadata" / "labels.json"
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            labels_path.write_text(json.dumps(labels[:1]), encoding="utf-8")
            self._write_registry(root)
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
                f"""schema_version: mmprism.csl_news_source_manifest.v2
source:
  data_root: {root}
  archive_root: /absolute/archives
  labels_path: incoming/labels.json
  integrity_registry: manifests/integrity/registry.json
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
