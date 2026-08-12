import csv
import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path

from mmprism.data import (
    INTAKE_RECORD_SCHEMA,
    CslDailyIntakeDestinationExistsError,
    CslDailyIntakeError,
    promote_csl_daily_batch,
    validate_csl_daily_batch,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CslDailyIntakeTest(unittest.TestCase):
    def _stage_batch(self, root: Path, *, with_annotation: bool = True) -> Path:
        batch = root / "incoming" / "20260812_csl_daily_fixture"
        payload = batch / "external_sources" / "csl_daily"
        images_a = payload / "sentence" / "images" / "S000000_P0004_T00"
        images_b = payload / "sentence" / "images" / "S000001_P0000_T00"
        images_a.mkdir(parents=True)
        images_b.mkdir(parents=True)

        files: dict[str, bytes] = {}
        for sequence_dir, frames in (
            (images_a, ("000000.jpg", "000001.jpg")),
            (images_b, ("000000.jpg",)),
        ):
            for frame in frames:
                content = f"jpeg-bytes-{sequence_dir.name}-{frame}".encode()
                (sequence_dir / frame).write_bytes(content)
                relative = (
                    sequence_dir / frame
                ).relative_to(batch).as_posix()
                files[relative] = content

        if with_annotation:
            annotation_dir = payload / "sentence_label"
            annotation_dir.mkdir(parents=True, exist_ok=True)
            annotation_path = annotation_dir / "csl2020ct_v2.pkl"
            with annotation_path.open("wb") as stream:
                pickle.dump(
                    {
                        "info": [
                            {
                                "name": "S000000_P0004_T00",
                                "label_char": ["我", "爱", "你"],
                            },
                            {
                                "name": "S000001_P0000_T00",
                                "label_char": ["谢", "谢"],
                            },
                        ]
                    },
                    stream,
                )
            files["external_sources/csl_daily/sentence_label/csl2020ct_v2.pkl"] = (
                annotation_path.read_bytes()
            )

        metadata_path = payload / "SOURCE_METADATA.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "dataset_name": "CSL-Daily",
                    "dataset_version": "csl2020ct_v2",
                    "download_date": "2026-08-01",
                    "source_url": "https://example.invalid/csl-daily",
                    "license": "research-only",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        files["external_sources/csl_daily/SOURCE_METADATA.json"] = (
            metadata_path.read_bytes()
        )

        (batch / "README.md").write_text(
            "CSL-Daily P0-D upload batch (fixture).\n", encoding="utf-8"
        )

        manifest_path = batch / "UPLOAD_MANIFEST.csv"
        with manifest_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "source_id",
                    "relative_path",
                    "category",
                    "dataset",
                    "size_bytes",
                    "sha256",
                    "source_owner",
                    "access_class",
                    "original_format",
                    "notes",
                ]
            )
            for relative, content in sorted(files.items()):
                writer.writerow(
                    [
                        "csl_daily_fixture",
                        relative,
                        "external_sources",
                        "csl_daily",
                        str(len(content)),
                        _sha256(content),
                        "fixture",
                        "research",
                        relative.rsplit(".", 1)[-1],
                        "",
                    ]
                )
        (batch / "SHA256SUMS").write_text(
            "".join(
                f"{_sha256(content)}  {relative}\n"
                for relative, content in sorted(files.items())
            ),
            encoding="utf-8",
        )
        return batch

    def test_validate_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = self._stage_batch(root)
            report = validate_csl_daily_batch(batch)
            self.assertTrue(report.ok, report.failed_checks())
            self.assertEqual(report.batch_id, "20260812_csl_daily_fixture")
            self.assertIsNotNone(report.metadata)
            assert report.metadata is not None
            self.assertEqual(report.metadata.dataset_version, "csl2020ct_v2")
            self.assertEqual(len(report.manifest_entries), 5)
            check_names = {check.name for check in report.checks}
            self.assertIn("image_layout", check_names)
            self.assertIn("annotation_present", check_names)

    def test_validate_rejects_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = self._stage_batch(root)
            tampered = (
                batch
                / "external_sources"
                / "csl_daily"
                / "sentence"
                / "images"
                / "S000000_P0004_T00"
                / "000000.jpg"
            )
            tampered.write_bytes(b"tampered")
            report = validate_csl_daily_batch(batch)
            self.assertFalse(report.ok)
            failed = {check.name for check in report.failed_checks()}
            self.assertIn("manifest_files_match", failed)
            self.assertIn("sha256sums_match", failed)
            with self.assertRaises(CslDailyIntakeError):
                promote_csl_daily_batch(batch, root)

    def test_validate_rejects_missing_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = self._stage_batch(root, with_annotation=False)
            report = validate_csl_daily_batch(batch)
            self.assertFalse(report.ok)
            failed = {check.name for check in report.failed_checks()}
            self.assertIn("annotation_present", failed)

    def test_promotion_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = self._stage_batch(root)
            report = validate_csl_daily_batch(batch)
            record = promote_csl_daily_batch(batch, root, report=report)

            destination = root / "raw" / "csl_daily"
            self.assertTrue((destination / "sentence_label" / "csl2020ct_v2.pkl").is_file())
            self.assertTrue(
                (
                    destination
                    / "sentence"
                    / "images"
                    / "S000000_P0004_T00"
                    / "000000.jpg"
                ).is_file()
            )
            self.assertEqual(record["schema_version"], INTAKE_RECORD_SCHEMA)
            self.assertEqual(record["batch"]["batch_id"], "20260812_csl_daily_fixture")
            self.assertEqual(
                record["source_metadata"]["download_date"], "2026-08-01"
            )
            self.assertEqual(record["verification"]["destination_file_count"], 5)
            inventory = {entry["relative_path"]: entry for entry in record["files"]}
            self.assertIn("sentence_label/csl2020ct_v2.pkl", inventory)

            record_path = destination / "INTAKE_RECORD.json"
            persisted = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["schema_version"], INTAKE_RECORD_SCHEMA)

            # Incoming batch stays untouched (copy, not move).
            self.assertTrue(
                (
                    batch
                    / "external_sources"
                    / "csl_daily"
                    / "sentence_label"
                    / "csl2020ct_v2.pkl"
                ).is_file()
            )
            # Destination checksums were re-verified and recorded.
            annotation_dest = destination / "sentence_label" / "csl2020ct_v2.pkl"
            self.assertEqual(
                _sha256(annotation_dest.read_bytes()),
                inventory["sentence_label/csl2020ct_v2.pkl"]["sha256"],
            )

    def test_promotion_refuses_to_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = self._stage_batch(root)
            promote_csl_daily_batch(batch, root)
            with self.assertRaises(CslDailyIntakeDestinationExistsError):
                promote_csl_daily_batch(batch, root)

    def test_promotion_rejects_failed_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = self._stage_batch(root, with_annotation=False)
            with self.assertRaises(CslDailyIntakeError):
                promote_csl_daily_batch(batch, root)
            self.assertFalse((root / "raw" / "csl_daily").exists())


if __name__ == "__main__":
    unittest.main()
