import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from mmprism.config import ConfigError, load_experiment_config
from mmprism.contracts import ManifestError, validate_manifest
from mmprism.data import (
    CslNewsAnnotationError,
    CslNewsAuditError,
    audit_csl_news_archive,
    build_csl_news_annotation_qc,
    build_csl_news_annotation_status,
    load_csl_news_annotation_config,
    run_csl_news_annotation,
    write_csl_news_annotation_qc,
    write_csl_news_annotation_status,
    write_csl_news_audit,
)
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

    csl_news_parser = subparsers.add_parser(
        "csl-news-audit", help="Audit one complete CSL-News archive against official labels"
    )
    csl_news_parser.add_argument("archive", type=Path)
    csl_news_parser.add_argument("--labels", type=Path, required=True)
    csl_news_parser.add_argument("--source-id", required=True)
    csl_news_parser.add_argument("--output", type=Path, required=True)
    csl_news_parser.add_argument("--decode-samples", type=int, default=0)
    csl_news_parser.add_argument("--scratch-dir", type=Path)
    csl_news_parser.add_argument("--skip-crc", action="store_true")

    annotation_parser = subparsers.add_parser(
        "csl-news-annotate", help="Build restartable RTMW3D annotations for CSL-News"
    )
    annotation_parser.add_argument("config", type=Path)
    annotation_parser.add_argument("--project-root", type=Path)
    annotation_parser.add_argument("--max-videos", type=int)
    annotation_parser.add_argument("--archive-id", type=int)
    annotation_parser.add_argument(
        "--once", action="store_true", help="Process currently complete archives and exit"
    )

    status_parser = subparsers.add_parser(
        "csl-news-annotation-status",
        help="Summarize CSL-News annotation progress and output integrity",
    )
    status_parser.add_argument("config", type=Path)
    status_parser.add_argument("--project-root", type=Path)
    status_parser.add_argument("--sample-validate", type=int, default=3)
    status_parser.add_argument("--recent-window", type=int, default=200)
    status_parser.add_argument("--output", type=Path)

    qc_parser = subparsers.add_parser(
        "csl-news-annotation-qc",
        help="Measure numerical quality on a deterministic annotation sample",
    )
    qc_parser.add_argument("config", type=Path)
    qc_parser.add_argument("--project-root", type=Path)
    qc_parser.add_argument("--sample-count", type=int, default=100)
    qc_parser.add_argument("--output", type=Path)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)

    exit_code = 0
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
        elif arguments.command == "manifest":
            payload = validate_manifest(arguments.path).to_dict()
        elif arguments.command == "csl-news-audit":
            payload = audit_csl_news_archive(
                arguments.archive,
                arguments.labels,
                source_id=arguments.source_id,
                verify_crc=not arguments.skip_crc,
                decode_sample_count=arguments.decode_samples,
                scratch_dir=arguments.scratch_dir,
            )
            write_csl_news_audit(payload, arguments.output)
            exit_code = 0 if payload["status"] == "passed" else 1
        elif arguments.command == "csl-news-annotate":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            annotation_config = load_csl_news_annotation_config(
                arguments.config, project_root
            )
            payload = run_csl_news_annotation(
                annotation_config,
                max_videos=arguments.max_videos,
                once=arguments.once,
                archive_id=arguments.archive_id,
            )
            exit_code = 0 if payload["failed"] == 0 else 1
        elif arguments.command == "csl-news-annotation-status":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            annotation_config = load_csl_news_annotation_config(
                arguments.config, project_root
            )
            payload = build_csl_news_annotation_status(
                annotation_config,
                sample_validate_count=arguments.sample_validate,
                recent_window=arguments.recent_window,
            )
            if arguments.output:
                write_csl_news_annotation_status(payload, arguments.output)
            exit_code = 0 if payload["status"] == "healthy" else 1
        else:
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            annotation_config = load_csl_news_annotation_config(
                arguments.config, project_root
            )
            payload = build_csl_news_annotation_qc(
                annotation_config, sample_count=arguments.sample_count
            )
            if arguments.output:
                write_csl_news_annotation_qc(payload, arguments.output)
            exit_code = 1 if payload["status"] == "failed" else 0
    except (
        ConfigError,
        ManifestError,
        CslNewsAnnotationError,
        CslNewsAuditError,
        FileNotFoundError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code
