import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mmprism.config import ConfigError, load_experiment_config


class ExperimentConfigTest(unittest.TestCase):
    def test_loads_and_resolves_environment_paths(self) -> None:
        payload = """\
schema_version: mmprism.experiment.v1
name: unit-test
task: pose_reconstruction
paths:
  data_root: ${MMPRISM_TEST_DATA}
  artifact_root: outputs
  cache_root: .cache/mmprism
runtime:
  seed: 7
  devices: [0, 2]
  precision: bf16-mixed
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(payload, encoding="utf-8")
            with patch.dict(os.environ, {"MMPRISM_TEST_DATA": "/tmp/mmprism-data"}):
                config = load_experiment_config(config_path).resolved(Path(directory))

        self.assertEqual(config.name, "unit-test")
        self.assertEqual(config.paths.data_root, Path("/tmp/mmprism-data"))
        self.assertEqual(config.runtime.devices, (0, 2))

    def test_rejects_unknown_keys(self) -> None:
        payload = """\
schema_version: mmprism.experiment.v1
name: invalid
task: evaluation
paths:
  data_root: data
  artifact_root: outputs
  cache_root: cache
surprise: true
"""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text(payload, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "Unknown keys"):
                load_experiment_config(config_path)


if __name__ == "__main__":
    unittest.main()
