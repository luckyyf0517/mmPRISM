import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime
from pathlib import Path

from mmprism.artifacts import ArtifactError, RunArtifactWriter, RunInput
from mmprism.cli import main
from mmprism.config import load_experiment_config
from mmprism.runtime import build_run_plan, discover_project_root


class RunArtifactWriterTest(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        config = root / "experiment.yaml"
        config.write_text(
            f"""\
schema_version: mmprism.experiment.v1
name: artifact-test
task: evaluation
paths:
  data_root: {root / 'data'}
  artifact_root: {root / 'artifacts'}
  cache_root: {root / 'cache'}
runtime:
  seed: 17
  devices: auto
  precision: 32-true
  deterministic: true
""",
            encoding="utf-8",
        )
        return config

    def test_initializes_writes_metrics_and_finalizes(self) -> None:
        project_root = discover_project_root(Path(__file__))
        created_at = datetime(2026, 8, 11, 18, 0, tzinfo=UTC)
        completed_at = datetime(2026, 8, 11, 18, 5, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._config(root)
            manifest = root / "manifest.jsonl"
            manifest.write_text('{"sample_id":"one"}\n', encoding="utf-8")
            config = load_experiment_config(config_path)
            plan = build_run_plan(config, project_root, created_at=created_at)
            run_input = RunInput.capture(
                name="data_manifest", kind="manifest", path=manifest
            )

            writer = RunArtifactWriter.initialize(
                plan,
                source_config=config_path,
                inputs=(run_input,),
                command=("mmprism", "evaluate"),
            )

            self.assertEqual(writer.run_dir, plan.run_dir)
            self.assertEqual(
                sorted(path.name for path in writer.run_dir.iterdir()),
                ["config.resolved.json", "environment.json", "inputs.json", "run.json"],
            )
            inputs = json.loads((writer.run_dir / "inputs.json").read_text(encoding="utf-8"))
            self.assertEqual(inputs["inputs"][0]["name"], "data_manifest")
            self.assertEqual(
                inputs["inputs"][0]["sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest()
            )

            writer.write_metrics(
                protocol_id="translation.v1",
                split="test",
                values={"bleu4": 0.25, "samples": 1},
                sample_count=1,
                created_at=completed_at,
            )
            writer.finalize(status="completed", completed_at=completed_at)

            run = json.loads((writer.run_dir / "run.json").read_text(encoding="utf-8"))
            metrics = json.loads((writer.run_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(run["status"], "completed")
            self.assertEqual(run["experiment"]["seed"], 17)
            self.assertIn("metrics.json", run["artifacts"])
            self.assertEqual(metrics["protocol_id"], "translation.v1")
            self.assertEqual(metrics["values"], {"bleu4": 0.25, "samples": 1})
            self.assertFalse(any(path.name.startswith(".") for path in writer.run_dir.iterdir()))

    def test_refuses_collisions_hash_mismatches_and_invalid_metrics(self) -> None:
        project_root = discover_project_root(Path(__file__))
        created_at = datetime(2026, 8, 11, 18, 0, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._config(root)
            manifest = root / "manifest.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ArtifactError, "SHA-256 mismatch"):
                RunInput.capture(
                    name="data_manifest",
                    kind="manifest",
                    path=manifest,
                    expected_sha256="0" * 64,
                )

            config = load_experiment_config(config_path)
            plan = build_run_plan(config, project_root, created_at=created_at)
            run_input = RunInput.capture(
                name="data_manifest", kind="manifest", path=manifest
            )
            with self.assertRaisesRegex(ArtifactError, "launch command"):
                RunArtifactWriter.initialize(
                    plan, source_config=config_path, inputs=(run_input,)
                )
            with self.assertRaisesRegex(ArtifactError, "registered inputs"):
                RunArtifactWriter.initialize(
                    plan, source_config=config_path, command=("mmprism", "evaluate")
                )
            with self.assertRaisesRegex(ArtifactError, "manifest input"):
                RunArtifactWriter.initialize(
                    plan,
                    source_config=config_path,
                    inputs=(RunInput.capture(name="aux", kind="other", path=manifest),),
                    command=("mmprism", "evaluate"),
                )
            writer = RunArtifactWriter.initialize(
                plan,
                source_config=config_path,
                inputs=(run_input,),
                command=("mmprism", "evaluate"),
            )
            with self.assertRaisesRegex(ArtifactError, "already exists"):
                RunArtifactWriter.initialize(
                    plan,
                    source_config=config_path,
                    inputs=(run_input,),
                    command=("mmprism", "evaluate"),
                )
            with self.assertRaisesRegex(ArtifactError, "finite"):
                writer.write_metrics(
                    protocol_id="metric.v1",
                    split="test",
                    values={"loss": float("nan")},
                    sample_count=1,
                )
            with self.assertRaisesRegex(ArtifactError, "require metrics"):
                writer.finalize(status="completed")
            writer.finalize(status="failed", failure="intentional test failure")

    def test_source_configuration_must_match_plan(self) -> None:
        project_root = discover_project_root(Path(__file__))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._config(root)
            manifest = root / "manifest.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            config = load_experiment_config(config_path)
            plan = build_run_plan(
                config,
                project_root,
                created_at=datetime(2026, 8, 11, 18, 0, tzinfo=UTC),
            )
            config_path.write_text(
                config_path.read_text(encoding="utf-8").replace("seed: 17", "seed: 18"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ArtifactError, "does not match"):
                RunArtifactWriter.initialize(
                    plan,
                    source_config=config_path,
                    inputs=(
                        RunInput.capture(
                            name="data_manifest", kind="manifest", path=manifest
                        ),
                    ),
                    command=("mmprism", "evaluate"),
                )

    def test_run_init_cli_registers_named_inputs(self) -> None:
        project_root = discover_project_root(Path(__file__))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = self._config(root)
            manifest = root / "manifest.jsonl"
            manifest.write_text('{"sample_id":"one"}\n', encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "run-init",
                        str(config_path),
                        "--project-root",
                        str(project_root),
                        "--input",
                        f"manifest:data_manifest={manifest}",
                    ]
                )

            self.assertEqual(exit_code, 0)
            result = json.loads(output.getvalue())
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "run.json").is_file())
            inputs = json.loads((run_dir / "inputs.json").read_text(encoding="utf-8"))
            self.assertEqual(inputs["inputs"][0]["kind"], "manifest")
            self.assertEqual(inputs["inputs"][0]["name"], "data_manifest")


if __name__ == "__main__":
    unittest.main()
