import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from mmprism.artifacts import ArtifactError, RunArtifactWriter, RunInput
from mmprism.assets import (
    ModelAssetError,
    download_model_assets,
    load_model_asset_config,
    plan_model_assets,
    run_model_asset_smoke,
    write_model_asset_smoke,
)
from mmprism.config import ConfigError, load_experiment_config
from mmprism.contracts import ManifestError, SplitContractError, validate_manifest
from mmprism.data import (
    CslNewsAnnotationError,
    CslNewsAuditError,
    CslNewsIntegrityError,
    CslNewsMetadataError,
    CslNewsPoseManifestError,
    CslNewsSourceManifestError,
    DataSplitError,
    audit_csl_news_archive,
    build_csl_news_annotation_qc,
    build_csl_news_annotation_status,
    build_csl_news_metadata_profile,
    build_csl_news_pose_manifest_snapshot,
    build_csl_news_source_manifest_snapshot,
    build_data_split_snapshot,
    load_csl_news_annotation_config,
    load_csl_news_integrity_config,
    load_csl_news_pose_manifest_config,
    load_csl_news_source_manifest_config,
    load_data_split_config,
    run_csl_news_annotation,
    scan_csl_news_source_integrity,
    write_csl_news_annotation_qc,
    write_csl_news_annotation_status,
    write_csl_news_audit,
    write_csl_news_metadata_profile,
)
from mmprism.release import (
    ReleaseAuditError,
    audit_release,
    load_release_audit_config,
    write_release_audit,
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

    run_init_parser = subparsers.add_parser(
        "run-init", help="Atomically initialize a formal run artifact directory"
    )
    run_init_parser.add_argument("path", type=Path)
    run_init_parser.add_argument("--project-root", type=Path)
    run_init_parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="KIND:NAME=PATH",
        help="Hash and register an immutable run input; may be repeated",
    )

    manifest_parser = subparsers.add_parser("manifest", help="Validate a JSONL data manifest")
    manifest_parser.add_argument("path", type=Path)

    split_parser = subparsers.add_parser(
        "split", help="Build an atomic deterministic group-disjoint split snapshot"
    )
    split_parser.add_argument("config", type=Path)
    split_parser.add_argument("--project-root", type=Path)
    split_parser.add_argument("--snapshot-id")

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
    annotation_parser.add_argument("--worker-index", type=int)
    annotation_parser.add_argument("--worker-count", type=int)
    annotation_parser.add_argument("--integrity-registry", type=Path)
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
    status_parser.add_argument("--integrity-registry", type=Path)

    qc_parser = subparsers.add_parser(
        "csl-news-annotation-qc",
        help="Measure numerical quality on a deterministic annotation sample",
    )
    qc_parser.add_argument("config", type=Path)
    qc_parser.add_argument("--project-root", type=Path)
    qc_parser.add_argument("--sample-count", type=int, default=100)
    qc_parser.add_argument("--output", type=Path)

    metadata_parser = subparsers.add_parser(
        "csl-news-metadata-profile",
        help="Validate and characterize pinned CSL-News label metadata",
    )
    metadata_parser.add_argument("--labels-json", type=Path, required=True)
    metadata_parser.add_argument("--labels-csv", type=Path, required=True)
    metadata_parser.add_argument("--dataset-card", type=Path, required=True)
    metadata_parser.add_argument("--source-id", required=True)
    metadata_parser.add_argument("--source-revision", required=True)
    metadata_parser.add_argument("--output", type=Path)

    source_manifest_parser = subparsers.add_parser(
        "csl-news-source-manifest",
        help="Build an atomic source manifest for currently complete CSL-News archives",
    )
    source_manifest_parser.add_argument("config", type=Path)
    source_manifest_parser.add_argument("--project-root", type=Path)

    pose_manifest_parser = subparsers.add_parser(
        "csl-news-pose-manifest",
        help="Freeze validated CSL-News pose and caption artifacts into a manifest",
    )
    pose_manifest_parser.add_argument("config", type=Path)
    pose_manifest_parser.add_argument("--project-root", type=Path)
    pose_manifest_parser.add_argument("--snapshot-id")

    integrity_parser = subparsers.add_parser(
        "csl-news-integrity-scan",
        help="Incrementally audit final CSL-News ZIPs into a cumulative registry",
    )
    integrity_parser.add_argument("config", type=Path)
    integrity_parser.add_argument("--project-root", type=Path)
    integrity_parser.add_argument("--max-new-archives", type=int)
    integrity_parser.add_argument("--archive-id", type=int)

    release_parser = subparsers.add_parser(
        "release-audit",
        help="Audit the public release inventory, imports, and entrypoints",
    )
    release_parser.add_argument("config", type=Path)
    release_parser.add_argument("--project-root", type=Path)
    release_parser.add_argument("--output", type=Path)

    models_plan_parser = subparsers.add_parser(
        "models-plan",
        help="Inspect pinned model assets without network access or writes",
    )
    models_plan_parser.add_argument("config", type=Path)
    models_plan_parser.add_argument("--project-root", type=Path)
    models_plan_parser.add_argument("--output-root", type=Path)

    models_download_parser = subparsers.add_parser(
        "models-download",
        help="Download and checksum pinned model assets atomically",
    )
    models_download_parser.add_argument("config", type=Path)
    models_download_parser.add_argument("--project-root", type=Path)
    models_download_parser.add_argument("--output-root", type=Path)

    models_smoke_parser = subparsers.add_parser(
        "models-smoke",
        help="Load pinned evaluator models and validate finite embeddings",
    )
    models_smoke_parser.add_argument("config", type=Path)
    models_smoke_parser.add_argument("--project-root", type=Path)
    models_smoke_parser.add_argument("--output-root", type=Path)
    models_smoke_parser.add_argument("--device", default="cpu")
    models_smoke_parser.add_argument("--output", type=Path)

    return parser


def _resolve_model_root(argument: Path | None, project_root: Path) -> Path:
    value: str | Path
    if argument is not None:
        value = argument
    else:
        value = os.environ.get("MMPRISM_MODEL_ROOT", "pretrained_models")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _capture_run_inputs(specifications: Sequence[str], project_root: Path) -> tuple[RunInput, ...]:
    inputs: list[RunInput] = []
    for specification in specifications:
        identity, separator, path_text = specification.partition("=")
        kind, kind_separator, name = identity.partition(":")
        if not separator or not kind_separator or not kind or not name or not path_text:
            raise ArtifactError(
                f"invalid --input {specification!r}; expected KIND:NAME=PATH"
            )
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = project_root / path
        inputs.append(RunInput.capture(name=name, kind=kind, path=path))
    return tuple(inputs)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    arguments = parser.parse_args(effective_argv)

    exit_code = 0
    try:
        if arguments.command == "doctor":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            payload = collect_runtime_report(project_root)
        elif arguments.command in {"config", "plan", "run-init"}:
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            config = load_experiment_config(arguments.path)
            if arguments.command == "config":
                payload = config.resolved(project_root).to_dict()
            else:
                plan = build_run_plan(config, project_root)
                if arguments.command == "plan":
                    payload = plan.to_dict()
                else:
                    inputs = _capture_run_inputs(arguments.input, project_root)
                    writer = RunArtifactWriter.initialize(
                        plan,
                        source_config=arguments.path,
                        inputs=inputs,
                        command=("mmprism", *effective_argv),
                    )
                    payload = {
                        "schema_version": "mmprism.run-init-result.v1",
                        "run_id": writer.run_id,
                        "run_dir": str(writer.run_dir),
                        "run_metadata": str(writer.run_dir / "run.json"),
                    }
        elif arguments.command == "manifest":
            payload = validate_manifest(arguments.path).to_dict()
        elif arguments.command == "split":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            split_config = load_data_split_config(arguments.config)
            payload = build_data_split_snapshot(
                split_config,
                runtime_report=collect_runtime_report(project_root),
                snapshot_id=arguments.snapshot_id,
            )
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
                worker_index=arguments.worker_index,
                worker_count=arguments.worker_count,
                integrity_registry_path=arguments.integrity_registry,
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
                integrity_registry_path=arguments.integrity_registry,
            )
            if arguments.output:
                write_csl_news_annotation_status(payload, arguments.output)
            exit_code = 0 if payload["status"] == "healthy" else 1
        elif arguments.command == "csl-news-annotation-qc":
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
        elif arguments.command == "csl-news-metadata-profile":
            payload = build_csl_news_metadata_profile(
                arguments.labels_json,
                arguments.labels_csv,
                arguments.dataset_card,
                source_id=arguments.source_id,
                source_revision=arguments.source_revision,
            )
            if arguments.output:
                write_csl_news_metadata_profile(payload, arguments.output)
            exit_code = 1 if payload["status"] == "failed" else 0
        elif arguments.command == "csl-news-source-manifest":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            source_manifest_config = load_csl_news_source_manifest_config(
                arguments.config
            )
            payload = build_csl_news_source_manifest_snapshot(
                source_manifest_config,
                runtime_report=collect_runtime_report(project_root),
            )
            exit_code = 0
        elif arguments.command == "csl-news-pose-manifest":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            pose_manifest_config = load_csl_news_pose_manifest_config(
                arguments.config
            )
            payload = build_csl_news_pose_manifest_snapshot(
                pose_manifest_config,
                runtime_report=collect_runtime_report(project_root),
                snapshot_id=arguments.snapshot_id,
            )
            exit_code = 0
        elif arguments.command == "csl-news-integrity-scan":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            integrity_config = load_csl_news_integrity_config(arguments.config)
            payload = scan_csl_news_source_integrity(
                integrity_config,
                runtime_report=collect_runtime_report(project_root),
                max_new_archives=arguments.max_new_archives,
                archive_id=arguments.archive_id,
            )
            exit_code = 0
        elif arguments.command == "release-audit":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            release_config = load_release_audit_config(arguments.config)
            payload = audit_release(release_config, project_root=project_root)
            if arguments.output:
                write_release_audit(payload, arguments.output)
            exit_code = 0 if payload["status"] == "passed" else 1
        else:
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            model_config = load_model_asset_config(arguments.config)
            model_root = _resolve_model_root(arguments.output_root, project_root)
            if arguments.command == "models-plan":
                payload = plan_model_assets(model_config, model_root)
                exit_code = 0 if payload["status"] == "ready" else 1
            elif arguments.command == "models-download":
                payload = download_model_assets(
                    model_config,
                    model_root,
                    runtime_report=collect_runtime_report(project_root),
                )
                exit_code = 0
            else:
                payload = run_model_asset_smoke(
                    model_config, model_root, device=arguments.device
                )
                if arguments.output:
                    write_model_asset_smoke(payload, arguments.output)
                exit_code = 0
    except (
        ArtifactError,
        ConfigError,
        ManifestError,
        ModelAssetError,
        ReleaseAuditError,
        CslNewsAnnotationError,
        CslNewsAuditError,
        CslNewsIntegrityError,
        CslNewsMetadataError,
        CslNewsPoseManifestError,
        CslNewsSourceManifestError,
        DataSplitError,
        FileNotFoundError,
        SplitContractError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code
