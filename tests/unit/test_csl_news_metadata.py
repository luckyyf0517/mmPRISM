import csv
import json
import tempfile
import unittest
from pathlib import Path

from mmprism.data import (
    CslNewsMetadataError,
    build_csl_news_metadata_profile,
    write_csl_news_metadata_profile,
)


class CslNewsMetadataProfileTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        records = [
            {
                "video": "Common-Concerns_20200101_0-100_1.mp4",
                "pose": "Common-Concerns_20200101_0-100_1.pkl",
                "text": "共同关注。",
            },
            {
                "video": "20200102_Dragon-TV__0-100_2.mp4",
                "pose": "20200102_Dragon-TV__0-100_2.pkl",
                "text": "共同关注。",
            },
        ]
        labels_json = root / "CSL_News_Labels.json"
        labels_json.write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8"
        )
        labels_csv = root / "CSL_News_Labels.csv"
        with labels_csv.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["video", "pose", "text"])
            writer.writeheader()
            writer.writerows(records)
        dataset_card = root / "README.md"
        dataset_card.write_text(
            """---
language:
- zh
license: cc-by-nc-4.0
task_categories:
- video-text-to-text
---
CSL-News is a Chinese Sign Language dataset.
""",
            encoding="utf-8",
        )
        return labels_json, labels_csv, dataset_card

    def test_profiles_translation_units_without_inventing_sign_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_fixture(Path(directory))
            report = build_csl_news_metadata_profile(
                *paths,
                source_id="fixture",
                source_revision="revision",
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["integrity"]["valid_record_count"], 2)
        self.assertTrue(report["integrity"]["csv_cross_check"]["exact_match"])
        self.assertEqual(report["dataset_units"]["translation_segment_count"], 2)
        self.assertEqual(report["dataset_units"]["unique_normalized_translation_count"], 1)
        self.assertIsNone(report["dataset_units"]["explicit_sentence_count"])
        self.assertIsNone(report["dataset_units"]["sign_vocabulary_size"])
        self.assertFalse(
            report["record_schema"]["explicit_field_availability"][
                "non_manual_features"
            ]
        )
        self.assertEqual(
            report["translation_statistics"]["length_units"]["han_codepoints"][
                "mean"
            ],
            4.0,
        )

    def test_fails_when_csv_and_json_differ(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_json, labels_csv, dataset_card = self._write_fixture(root)
            rows = list(csv.DictReader(labels_csv.read_text(encoding="utf-8").splitlines()))
            rows[1]["text"] = "不同文本。"
            with labels_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["video", "pose", "text"])
                writer.writeheader()
                writer.writerows(rows)
            report = build_csl_news_metadata_profile(
                labels_json,
                labels_csv,
                dataset_card,
                source_id="fixture",
                source_revision="revision",
            )

        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["integrity"]["csv_cross_check"]["exact_match"])

    def test_warns_when_csv_adds_conflicting_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_json, labels_csv, dataset_card = self._write_fixture(root)
            rows = list(csv.DictReader(labels_csv.read_text(encoding="utf-8").splitlines()))
            duplicate = dict(rows[0])
            duplicate["text"] = "冲突文本。"
            with labels_csv.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["video", "pose", "text"])
                writer.writeheader()
                writer.writerows([*rows, duplicate])
            report = build_csl_news_metadata_profile(
                labels_json,
                labels_csv,
                dataset_card,
                source_id="fixture",
                source_revision="revision",
            )

        cross_check = report["integrity"]["csv_cross_check"]
        self.assertEqual(report["status"], "passed_with_warnings")
        self.assertEqual(cross_check["canonical_json_record_missing_count"], 0)
        self.assertEqual(cross_check["duplicate_video_key_count"], 1)
        self.assertEqual(cross_check["conflicting_content_row_count"], 1)

    def test_rejects_partial_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_json, labels_csv, dataset_card = self._write_fixture(root)
            partial = root / "CSL_News_Labels.json.part"
            partial.write_bytes(labels_json.read_bytes())
            with self.assertRaisesRegex(CslNewsMetadataError, "must be complete"):
                build_csl_news_metadata_profile(
                    partial,
                    labels_csv,
                    dataset_card,
                    source_id="fixture",
                    source_revision="revision",
                )

    def test_writes_profile_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports" / "profile.json"
            written = write_csl_news_metadata_profile({"status": "passed"}, output)
            payload = json.loads(written.read_text(encoding="utf-8"))

        self.assertEqual(payload, {"status": "passed"})


if __name__ == "__main__":
    unittest.main()
