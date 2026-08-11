import json
import tempfile
import unittest
from pathlib import Path

from mmprism.contracts import ManifestError, SampleRecord, validate_manifest


def sample(sample_id: str) -> dict[str, object]:
    return {
        "schema_version": "mmprism.sample.v1",
        "sample_id": sample_id,
        "sequence_id": "sequence-1",
        "subject_id": "subject-1",
        "dataset": "fixture",
        "modalities": {
            "radar_cube": {
                "uri": f"processed/{sample_id}.npy",
                "shape": [64, 32, 32, 32],
                "dtype": "float32",
            }
        },
        "group_keys": {"subject": "subject-1"},
    }


class ManifestContractTest(unittest.TestCase):
    def test_validates_manifest_and_summarizes_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            records = [sample("sample-1"), sample("sample-2")]
            path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            summary = validate_manifest(path)

        self.assertEqual(summary.record_count, 2)
        self.assertEqual(summary.datasets, ("fixture",))
        self.assertEqual(summary.modalities, ("radar_cube",))

    def test_rejects_duplicate_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            record = json.dumps(sample("duplicate"))
            path.write_text(f"{record}\n{record}\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestError, "Duplicate sample_id"):
                validate_manifest(path)

    def test_rejects_absolute_local_modality_paths(self) -> None:
        record = sample("sample-1")
        record["modalities"] = {"radar_cube": {"uri": "/private/machine/data.npy"}}
        with self.assertRaisesRegex(ManifestError, "must be relative"):
            SampleRecord.from_mapping(record)

    def test_accepts_exactly_one_inline_text_payload(self) -> None:
        record = sample("sample-1")
        record["modalities"] = {
            "caption": {
                "text": "测试文本",
                "dtype": "utf-8",
                "sha256": "a" * 64,
            }
        }
        parsed = SampleRecord.from_mapping(record)

        self.assertEqual(parsed.modalities["caption"].text, "测试文本")
        self.assertIsNone(parsed.modalities["caption"].uri)

    def test_rejects_modality_with_uri_and_inline_text(self) -> None:
        record = sample("sample-1")
        record["modalities"] = {
            "caption": {"uri": "labels/sample.txt", "text": "ambiguous"}
        }
        with self.assertRaisesRegex(ManifestError, "exactly one"):
            SampleRecord.from_mapping(record)


if __name__ == "__main__":
    unittest.main()
