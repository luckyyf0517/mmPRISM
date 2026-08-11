import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mmprism.config import ConfigError, load_experiment_config
from mmprism.contracts import ManifestError, validate_manifest
from mmprism.runtime import build_run_plan, collect_runtime_report, discover_project_root


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmprism")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Report the current runtime state")
    doctor_parser.add_argument("--project-root", type=Path)

    config_parser = subparsers.add_parser("config", help="Validate an experiment configuration")
    config_parser.add_argument("path", type=Path)
    config_parser.add_argument("--project-root", type=Path)

    plan_parser = subparsers.add_parser("plan", help="Resolve a side-effect-free run plan")
    plan_parser.add_argument("path", type=Path)
    plan_parser.add_argument("--project-root", type=Path)

    manifest_parser = subparsers.add_parser("manifest", help="Validate a JSONL data manifest")
    manifest_parser.add_argument("path", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "doctor":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            payload = collect_runtime_report(project_root)
        elif arguments.command in {"config", "plan"}:
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            config = load_experiment_config(arguments.path)
            payload = (
                config.resolved(project_root).to_dict()
                if arguments.command == "config"
                else build_run_plan(config, project_root).to_dict()
            )
        else:
            payload = validate_manifest(arguments.path).to_dict()
    except (ConfigError, ManifestError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
