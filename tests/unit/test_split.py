import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from mmprism.cli import main
from mmprism.contracts import (
    SplitContractError,
    SplitIndex,
    validate_split_assignments,
)
from mmprism.data import (
    DataSplitConfig,
    DataSplitError,
    build_data_split_snapshot,
    load_data_split_config,
)


class DataSplitTest(unittest.TestCase):
    def _manifest(self, root: Path, *, group_count: int = 100) -> Path:
        path = root / "manifests" / "source" / "manifest.jsonl"
        path.parent.mkdir(parents=True)
        records = []
        for group_index in range(group_count):
            for member_index in range(2):
                sample_id = f"sample-{group_index:04d}-{member_index}"
                records.append(
                    {
                        "schema_version": "mmprism.sample.v1",
                        "sample_id": sample_id,
                        "sequence_id": f"sequence-{group_index:04d}",
                        "dataset": "fixture",
                        "modalities": {
                            "caption": {
                                "text": sample_id,
                                "dtype": "utf-8",
                            }
                        },
                    }
                )
        path.write_text(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )
        return path

    def _config(self, root: Path, manifest_path: Path) -> DataSplitConfig:
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        config_path = root / "split.yaml"
        config_path.write_text(
            f"""schema_version: mmprism.split_config.v1
source:
  data_root: ${{MMPRISM_TEST_DATA_ROOT}}
  manifest: manifests/source/manifest.jsonl
  expected_manifest_sha256: {manifest_sha256}
  expected_dataset: fixture
  expected_record_count: 200
  scope: partial
grouping:
  selector: sequence_id
  namespace: fixture_sequence_v1
assignment:
  algorithm: sha256_mod_weight_v1
  protocol_id: fixture_80_10_10_v1
  seed: 17
  splits:
    - name: train
      weight: 8
    - name: validation
      weight: 1
    - name: test
      weight: 1
validation:
  minimum_groups_per_split: 1
  minimum_free_bytes: 0
output:
  snapshot_root: splits/fixture
""",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"MMPRISM_TEST_DATA_ROOT": str(root)}):
            return load_data_split_config(config_path)

    def _runtime(self, *, dirty: bool = False) -> dict[str, object]:
        return {"git": {"commit": "a" * 40, "dirty": dirty}, "python": "3.12"}

    def test_builds_reproducible_group_disjoint_snapshot_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._manifest(root)
            config = self._config(root, manifest_path)
            first = build_data_split_snapshot(
                config, runtime_report=self._runtime(), snapshot_id="first"
            )
            second = build_data_split_snapshot(
                config, runtime_report=self._runtime(), snapshot_id="second"
            )
            first_path = Path(first["assignments_path"])
            second_path = Path(second["assignments_path"])
            first_bytes = first_path.read_bytes()
            second_bytes = second_path.read_bytes()
            summary = json.loads(Path(first["summary_path"]).read_text())
            validation = validate_split_assignments(first_path)
            index = SplitIndex(first_path)

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first["assignment_count"], 200)
        self.assertEqual(first["group_count"], 100)
        self.assertEqual(sum(first["sample_counts"].values()), 200)
        self.assertEqual(sum(first["group_counts"].values()), 100)
        self.assertEqual(validation.splits, ("test", "train", "validation"))
        self.assertEqual(index.splits, validation.splits)
        self.assertEqual(len(index), 200)
        self.assertEqual(
            index["sample-0000-0"].group_id,
            index["sample-0000-1"].group_id,
        )
        self.assertEqual(
            index["sample-0000-0"].split,
            index["sample-0000-1"].split,
        )
        self.assertNotIn(str(root), first_bytes.decode())
        self.assertEqual(summary["audit"]["cross_split_group_leakage_count"], 0)
        self.assertEqual(summary["source"]["manifest_sha256"], config.expected_manifest_sha256)

    def test_rejects_dirty_git_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._manifest(root)
            config = self._config(root, manifest_path)
            with self.assertRaisesRegex(DataSplitError, "clean Git"):
                build_data_split_snapshot(
                    config,
                    runtime_report=self._runtime(dirty=True),
                    snapshot_id="dirty",
                )

    def test_cli_builds_the_same_split_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._manifest(root)
            self._config(root, manifest_path)
            output = io.StringIO()
            with (
                patch.dict(os.environ, {"MMPRISM_TEST_DATA_ROOT": str(root)}),
                patch("mmprism.cli.collect_runtime_report", return_value=self._runtime()),
                redirect_stdout(output),
            ):
                exit_code = main(
                    [
                        "split",
                        str(root / "split.yaml"),
                        "--project-root",
                        str(root),
                        "--snapshot-id",
                        "cli",
                    ]
                )

            result = json.loads(output.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(result["assignment_count"], 200)
            self.assertTrue(Path(result["assignments_path"]).is_file())

    def test_rejects_source_manifest_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = self._manifest(root)
            config = self._config(root, manifest_path)
            manifest_path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(DataSplitError, "SHA-256 mismatch"):
                build_data_split_snapshot(
                    config,
                    runtime_report=self._runtime(),
                    snapshot_id="checksum",
                )

    def test_rejects_absolute_storage_relationship(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "invalid.yaml"
            config_path.write_text(
                f"""schema_version: mmprism.split_config.v1
source:
  data_root: {root}
  manifest: /absolute/manifest.jsonl
  expected_manifest_sha256: {'a' * 64}
  expected_dataset: fixture
  expected_record_count: 1
  scope: partial
grouping:
  selector: sequence_id
  namespace: fixture
assignment:
  algorithm: sha256_mod_weight_v1
  protocol_id: fixture
  seed: 1
  splits:
    - name: train
      weight: 1
    - name: test
      weight: 1
validation:
  minimum_groups_per_split: 1
  minimum_free_bytes: 0
output:
  snapshot_root: splits
""",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DataSplitError, "must be relative"):
                load_data_split_config(config_path)

    def test_validator_rejects_cross_split_group_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignments.jsonl"
            records = [
                {
                    "schema_version": "mmprism.split_assignment.v1",
                    "sample_id": "sample-1",
                    "group_id": "a" * 64,
                    "split": "train",
                },
                {
                    "schema_version": "mmprism.split_assignment.v1",
                    "sample_id": "sample-2",
                    "group_id": "a" * 64,
                    "split": "test",
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SplitContractError, "Group leakage"):
                validate_split_assignments(path)


if __name__ == "__main__":
    unittest.main()
