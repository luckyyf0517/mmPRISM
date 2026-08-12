import json
import pickle
import tempfile
import unittest
from pathlib import Path

from mmprism.data import (
    ANNOTATION_JSONL_SCHEMA,
    CslDailyAnnotationError,
    build_csl_daily_annotation_jsonl,
    load_csl_daily_annotations,
    write_csl_daily_annotation_jsonl,
)


class CslDailyAnnotationTest(unittest.TestCase):
    def _write_pickle(self, root: Path, payload: object) -> Path:
        path = root / "csl2020ct_v2.pkl"
        with path.open("wb") as stream:
            pickle.dump(payload, stream)
        return path

    def _fixture_payload(self) -> dict[str, object]:
        return {
            "info": [
                {
                    "name": "S000000_P0004_T00",
                    "label_char": ["我", "爱", "你"],
                    "label_gloss": ["我", "爱", "你"],
                },
                {
                    "name": "S000001_P0000_T00",
                    "label_char": ["谢", "谢"],
                    "label_gloss": [{"gloss": "谢谢"}],
                },
                {
                    "name": "S000002_P0008_T00",
                    "label_char": ["早", "上", "好"],
                },
            ]
        }

    def test_loads_typed_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_pickle(Path(directory), self._fixture_payload())
            records = load_csl_daily_annotations(path)

            self.assertEqual(len(records), 3)
            first = records[0]
            self.assertEqual(first.name, "S000000_P0004_T00")
            self.assertEqual(first.caption, "我爱你")
            self.assertEqual(first.label_char, ("我", "爱", "你"))
            self.assertEqual(first.label_gloss, ("我", "爱", "你"))
            # Mapping-style gloss entries are normalized to plain strings.
            self.assertEqual(records[1].label_gloss, ("谢谢",))
            # Missing gloss labels are optional.
            self.assertEqual(records[2].label_gloss, ())
            self.assertEqual(records[2].caption, "早上好")

    def test_jsonl_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_pickle(root, self._fixture_payload())
            records = load_csl_daily_annotations(path)

            text = build_csl_daily_annotation_jsonl(records)
            lines = text.splitlines()
            self.assertEqual(len(lines), 3)
            decoded = [json.loads(line) for line in lines]
            self.assertEqual(decoded[0]["schema_version"], ANNOTATION_JSONL_SCHEMA)
            self.assertEqual(decoded[0]["name"], "S000000_P0004_T00")
            self.assertEqual(decoded[0]["caption"], "我爱你")
            self.assertEqual(decoded[0]["label_char"], ["我", "爱", "你"])
            self.assertEqual(decoded[1]["label_gloss"], ["谢谢"])
            self.assertEqual(decoded[2]["label_gloss"], [])

            output = write_csl_daily_annotation_jsonl(records, root / "out" / "annotations.jsonl")
            reread = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(reread, decoded)

    def test_missing_file_raises(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaises(CslDailyAnnotationError),
        ):
            load_csl_daily_annotations(Path(directory) / "absent.pkl")

    def test_duplicate_names_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = {
                "info": [
                    {"name": "S000000_P0004_T00", "label_char": ["好"]},
                    {"name": "S000000_P0004_T00", "label_char": ["坏"]},
                ]
            }
            path = self._write_pickle(Path(directory), payload)
            with self.assertRaises(CslDailyAnnotationError):
                load_csl_daily_annotations(path)

    def test_missing_label_char_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = {"info": [{"name": "S000000_P0004_T00"}]}
            path = self._write_pickle(Path(directory), payload)
            with self.assertRaises(CslDailyAnnotationError):
                load_csl_daily_annotations(path)

    def test_info_must_be_a_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_pickle(Path(directory), {"info": "not-a-list"})
            with self.assertRaises(CslDailyAnnotationError):
                load_csl_daily_annotations(path)


if __name__ == "__main__":
    unittest.main()
