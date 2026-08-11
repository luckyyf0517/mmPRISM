import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from mmprism.data import CslNewsAuditError, audit_csl_news_archive, write_csl_news_audit


class CslNewsAuditTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> tuple[Path, Path]:
        archive_path = root / "archive_001.zip"
        video_names = [
            "Common-Concerns_20200101_0-100_1.mp4",
            "20200102_Dragon-TV__0-100_2.mp4",
        ]
        with zipfile.ZipFile(archive_path, "w") as archive:
            for video_name in video_names:
                archive.writestr(video_name, b"not-a-real-video")

        labels_path = root / "CSL_News_Labels.json"
        labels_path.write_text(
            json.dumps(
                [
                    {"video": video_names[0], "pose": "first.pkl", "text": "第一条"},
                    {"video": video_names[1], "pose": "second.pkl", "text": "第二条"},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return archive_path, labels_path

    def test_audits_archive_and_label_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path, labels_path = self._write_fixture(root)
            report = audit_csl_news_archive(
                archive_path,
                labels_path,
                source_id="fixture@revision",
                decode_sample_count=0,
            )

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["archive"]["video_count"], 2)
        self.assertEqual(
            report["archive"]["program_counts"],
            {"Common-Concerns": 1, "Dragon-TV": 1},
        )
        self.assertEqual(report["coverage"]["missing_label_count"], 0)
        self.assertEqual(len(report["archive"]["sha256"]), 64)

    def test_reports_missing_labels_without_mutating_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path, labels_path = self._write_fixture(root)
            labels_path.write_text("[]", encoding="utf-8")
            original_archive = archive_path.read_bytes()
            report = audit_csl_news_archive(
                archive_path,
                labels_path,
                source_id="fixture@revision",
                decode_sample_count=0,
            )

            self.assertEqual(archive_path.read_bytes(), original_archive)

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["coverage"]["missing_label_count"], 2)

    def test_rejects_partial_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial_archive = root / "archive_001.zip.part"
            partial_archive.write_bytes(b"partial")
            labels_path = root / "CSL_News_Labels.json"
            labels_path.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(CslNewsAuditError, "must be complete"):
                audit_csl_news_archive(
                    partial_archive,
                    labels_path,
                    source_id="fixture@revision",
                )

    def test_writes_report_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "artifacts" / "report.json"
            written = write_csl_news_audit({"status": "passed"}, output_path)
            payload = json.loads(written.read_text(encoding="utf-8"))

        self.assertEqual(payload, {"status": "passed"})


if __name__ == "__main__":
    unittest.main()
