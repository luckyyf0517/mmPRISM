import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from mmprism.config.schema import ExperimentConfig
from mmprism.runtime import build_run_plan, discover_project_root


class RunPlanTest(unittest.TestCase):
    def test_builds_stable_side_effect_free_plan(self) -> None:
        root = discover_project_root(Path(__file__))
        created_at = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as directory:
            config = ExperimentConfig.from_mapping(
                {
                    "schema_version": "mmprism.experiment.v1",
                    "name": "run-plan-test",
                    "task": "evaluation",
                    "paths": {
                        "data_root": f"{directory}/data",
                        "artifact_root": f"{directory}/artifacts",
                        "cache_root": f"{directory}/cache",
                    },
                }
            )
            first = build_run_plan(config, root, created_at=created_at)
            second = build_run_plan(config, root, created_at=created_at)

            self.assertEqual(first.config_sha256, second.config_sha256)
            self.assertEqual(first.run_id, second.run_id)
            self.assertEqual(first.created_at, "2026-08-11T12:00:00Z")
            self.assertTrue(first.run_dir.is_absolute())
            self.assertFalse(first.run_dir.exists())
            self.assertIn("inputs.json", first.expected_artifacts)
            self.assertIn("predictions.jsonl", first.expected_artifacts)

    def test_rejects_naive_timestamp(self) -> None:
        root = discover_project_root(Path(__file__))
        config = ExperimentConfig.from_mapping(
            {
                "schema_version": "mmprism.experiment.v1",
                "name": "naive-time-test",
                "task": "evaluation",
                "paths": {
                    "data_root": "data",
                    "artifact_root": "artifacts",
                    "cache_root": "cache",
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            build_run_plan(config, root, created_at=datetime(2026, 8, 11, 12, 0))


if __name__ == "__main__":
    unittest.main()
