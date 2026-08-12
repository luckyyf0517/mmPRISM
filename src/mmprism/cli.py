import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from mmprism.artifacts import (
    ArtifactError,
    PrepareError,
    RunArtifactWriter,
    RunInput,
    build_prepare_report,
)
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
    ParquetDeliveryError,
    audit_csl_news_archive,
    build_csl_news_annotation_identity_audit,
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
    load_parquet_delivery_config,
    materialize_parquet_delivery,
    plan_parquet_delivery,
    run_csl_news_annotation,
    scan_csl_news_source_integrity,
    select_csl_news_integrity_archive,
    validate_parquet_delivery,
    write_csl_news_annotation_identity_audit,
    write_csl_news_annotation_qc,
    write_csl_news_annotation_status,
    write_csl_news_audit,
    write_csl_news_metadata_profile,
)
from mmprism.data.csl_news_scheduler import (
    CslNewsSchedulerError,
    build_csl_news_scheduler_status,
    initialize_csl_news_scheduler,
    run_csl_news_annotation_scheduled_worker,
    set_csl_news_scheduler_state,
)
from mmprism.release import (
    ReleaseAuditError,
    audit_release,
    load_release_audit_config,
    write_release_audit,
)
from mmprism.runtime import build_run_plan, collect_runtime_report, discover_project_root
from mmprism.training import (
    MT5SmokeError,
    OmniHandRunError,
    OmniHandSmokeError,
    WaveLLMRunError,
    load_mt5_smoke_config,
    load_omnihand_run_config,
    load_omnihand_smoke_config,
    load_wavellm_run_config,
)


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

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Validate a formal run and its manifest/split bindings without writes",
    )
    prepare_parser.add_argument("path", type=Path)
    prepare_parser.add_argument("--project-root", type=Path)
    prepare_parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="KIND:NAME=PATH",
        help="Hash and validate a formal input; may be repeated",
    )
    prepare_parser.add_argument(
        "--split-binding",
        action="append",
        required=True,
        metavar="MANIFEST_NAME=SPLIT",
        help="Bind every manifest input to its declared split; may be repeated",
    )

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

    parquet_plan_parser = subparsers.add_parser(
        "parquet-delivery-plan",
        help="Validate frozen data inputs and plan Parquet delivery without writes",
    )
    parquet_plan_parser.add_argument("config", type=Path)
    parquet_plan_parser.add_argument("--project-root", type=Path)

    parquet_build_parser = subparsers.add_parser(
        "parquet-delivery-build",
        help="Materialize an immutable Parquet delivery from frozen data inputs",
    )
    parquet_build_parser.add_argument("config", type=Path)
    parquet_build_parser.add_argument("--project-root", type=Path)

    parquet_validate_parser = subparsers.add_parser(
        "parquet-delivery-validate",
        help="Validate a completed immutable Parquet delivery",
    )
    parquet_validate_parser.add_argument("root", type=Path)
    parquet_validate_parser.add_argument("--skip-checksums", action="store_true")

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

    scheduler_init_parser = subparsers.add_parser(
        "csl-news-scheduler-init",
        help="Initialize the paused durable control plane for elastic CSL-News annotation workers",
    )
    scheduler_init_parser.add_argument("config", type=Path)
    scheduler_init_parser.add_argument("--project-root", type=Path)
    scheduler_init_parser.add_argument("--lease-seconds", type=int, default=900)

    for command, help_text in (
        ("csl-news-scheduler-pause", "Cooperatively pause elastic CSL-News annotation workers"),
        ("csl-news-scheduler-resume", "Resume elastic CSL-News annotation workers"),
    ):
        control_parser = subparsers.add_parser(command, help=help_text)
        control_parser.add_argument("config", type=Path)
        control_parser.add_argument("--project-root", type=Path)
        control_parser.add_argument("--reason")

    scheduler_status_parser = subparsers.add_parser(
        "csl-news-scheduler-status",
        help="Report elastic CSL-News annotation queue state and active leases",
    )
    scheduler_status_parser.add_argument("config", type=Path)
    scheduler_status_parser.add_argument("--project-root", type=Path)
    scheduler_status_parser.add_argument("--integrity-registry", type=Path, required=True)

    scheduled_worker_parser = subparsers.add_parser(
        "csl-news-annotate-scheduled",
        help="Run one elastic, lease-controlled CSL-News annotation worker",
    )
    scheduled_worker_parser.add_argument("config", type=Path)
    scheduled_worker_parser.add_argument("--project-root", type=Path)
    scheduled_worker_parser.add_argument("--integrity-registry", type=Path, required=True)
    scheduled_worker_parser.add_argument("--worker-id")
    scheduled_worker_parser.add_argument("--poll-seconds", type=int)
    scheduled_worker_parser.add_argument(
        "--once", action="store_true", help="Process at most one claimed archive and exit"
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

    identity_audit_parser = subparsers.add_parser(
        "csl-news-annotation-audit",
        help="Stream-audit every CSL-News pose artifact identity visible at start",
    )
    identity_audit_parser.add_argument("config", type=Path)
    identity_audit_parser.add_argument("--project-root", type=Path)
    identity_audit_parser.add_argument("--output", type=Path)

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
        help="Freeze registry-passed CSL-News archives into a source manifest",
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

    integrity_select_parser = subparsers.add_parser(
        "csl-news-integrity-select",
        help="Select and revalidate one registry-passed CSL-News archive",
    )
    integrity_select_parser.add_argument("config", type=Path)
    integrity_select_parser.add_argument("--archive-id", type=int)
    integrity_select_parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Validate path and stat only; intended for cheap status probes",
    )

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

    mt5_smoke_parser = subparsers.add_parser(
        "mt5-smoke",
        help="Run the pinned mT5 train-and-generate vertical smoke",
    )
    mt5_smoke_parser.add_argument("config", type=Path)
    mt5_smoke_parser.add_argument("--model-assets", type=Path, required=True)
    mt5_smoke_parser.add_argument("--project-root", type=Path)
    mt5_smoke_parser.add_argument("--model-root", type=Path)
    mt5_smoke_parser.add_argument("--device", default="auto")
    mt5_smoke_parser.add_argument("--output", type=Path, required=True)

    omnihand_smoke_parser = subparsers.add_parser(
        "omnihand-smoke",
        help="Run the canonical CubeNet pose-reconstruction vertical smoke",
    )
    omnihand_smoke_parser.add_argument("config", type=Path)
    omnihand_smoke_parser.add_argument("--project-root", type=Path)
    omnihand_smoke_parser.add_argument("--device", default="auto")
    omnihand_smoke_parser.add_argument("--output", type=Path, required=True)

    omnihand_train_parser = subparsers.add_parser(
        "omnihand-train",
        help="Train CubeNet from manifest-bound radar cubes and metric poses",
    )
    omnihand_train_parser.add_argument("experiment_config", type=Path)
    omnihand_train_parser.add_argument("task_config", type=Path)
    omnihand_train_parser.add_argument("--train-manifest", type=Path, required=True)
    omnihand_train_parser.add_argument("--validation-manifest", type=Path, required=True)
    omnihand_train_parser.add_argument("--split-assignments", type=Path, required=True)
    omnihand_train_parser.add_argument("--resume-state-metadata", type=Path)
    omnihand_train_parser.add_argument("--resume-state-tensors", type=Path)
    omnihand_train_parser.add_argument("--project-root", type=Path)

    omnihand_evaluate_parser = subparsers.add_parser(
        "omnihand-evaluate",
        help="Evaluate a checksum-bound CubeNet checkpoint and write predictions",
    )
    omnihand_evaluate_parser.add_argument("experiment_config", type=Path)
    omnihand_evaluate_parser.add_argument("task_config", type=Path)
    omnihand_evaluate_parser.add_argument("--manifest", type=Path, required=True)
    omnihand_evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    omnihand_evaluate_parser.add_argument("--checkpoint-metadata", type=Path, required=True)
    omnihand_evaluate_parser.add_argument("--split-assignments", type=Path, required=True)
    omnihand_evaluate_parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    omnihand_evaluate_parser.add_argument("--project-root", type=Path)

    wavellm_train_parser = subparsers.add_parser(
        "wavellm-train",
        help="Train geometry-guided mT5 from manifest-bound pose, confidence, and radar features",
    )
    wavellm_train_parser.add_argument("experiment_config", type=Path)
    wavellm_train_parser.add_argument("task_config", type=Path)
    wavellm_train_parser.add_argument("--model-assets", type=Path, required=True)
    wavellm_train_parser.add_argument("--model-root", type=Path)
    wavellm_train_parser.add_argument("--train-manifest", type=Path, required=True)
    wavellm_train_parser.add_argument("--validation-manifest", type=Path, required=True)
    wavellm_train_parser.add_argument("--split-assignments", type=Path, required=True)
    wavellm_train_parser.add_argument("--resume-state-metadata", type=Path)
    wavellm_train_parser.add_argument("--resume-state-tensors", type=Path)
    wavellm_train_parser.add_argument("--project-root", type=Path)

    wavellm_evaluate_parser = subparsers.add_parser(
        "wavellm-evaluate",
        help="Evaluate a checksum-bound geometry-guided mT5 checkpoint",
    )
    wavellm_evaluate_parser.add_argument("experiment_config", type=Path)
    wavellm_evaluate_parser.add_argument("task_config", type=Path)
    wavellm_evaluate_parser.add_argument("--model-assets", type=Path, required=True)
    wavellm_evaluate_parser.add_argument("--model-root", type=Path)
    wavellm_evaluate_parser.add_argument("--manifest", type=Path, required=True)
    wavellm_evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    wavellm_evaluate_parser.add_argument("--checkpoint-metadata", type=Path, required=True)
    wavellm_evaluate_parser.add_argument("--split-assignments", type=Path, required=True)
    wavellm_evaluate_parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    wavellm_evaluate_parser.add_argument("--project-root", type=Path)

    return parser


def _resolve_model_root(argument: Path | None, project_root: Path) -> Path:
    value: str | Path
    if argument is not None:
        value = argument
    else:
        value = os.environ.get("MMPRISM_MODEL_ROOT", "pretrained_models")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _resolve_mt5_model_root(argument: Path | None, project_root: Path) -> Path:
    value: str | Path
    if argument is not None:
        value = argument
    else:
        value = os.environ.get(
            "MMPRISM_MT5_MODEL_ROOT",
            os.environ.get("MMPRISM_MODEL_ROOT", "pretrained_models/mt5_base_v1"),
        )
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _capture_run_inputs(specifications: Sequence[str], project_root: Path) -> tuple[RunInput, ...]:
    inputs: list[RunInput] = []
    for specification in specifications:
        identity, separator, path_text = specification.partition("=")
        kind, kind_separator, name = identity.partition(":")
        if not separator or not kind_separator or not kind or not name or not path_text:
            raise ArtifactError(f"invalid --input {specification!r}; expected KIND:NAME=PATH")
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            path = project_root / path
        inputs.append(RunInput.capture(name=name, kind=kind, path=path))
    return tuple(inputs)


def _parse_split_bindings(specifications: Sequence[str]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for specification in specifications:
        name, separator, split = specification.partition("=")
        if not separator or not name or not split:
            raise PrepareError(
                f"invalid --split-binding {specification!r}; expected MANIFEST_NAME=SPLIT"
            )
        if name in bindings:
            raise PrepareError(f"duplicate split binding for manifest input {name!r}")
        bindings[name] = split
    return bindings


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
        elif arguments.command in {"config", "plan", "prepare", "run-init"}:
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            config = load_experiment_config(arguments.path)
            if arguments.command == "config":
                payload = config.resolved(project_root).to_dict()
            elif arguments.command == "prepare":
                inputs = _capture_run_inputs(arguments.input, project_root)
                payload = build_prepare_report(
                    config,
                    source_config=arguments.path,
                    inputs=inputs,
                    split_bindings=_parse_split_bindings(arguments.split_binding),
                    project_root=project_root,
                )
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
        elif arguments.command in {"parquet-delivery-plan", "parquet-delivery-build"}:
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            delivery_config = load_parquet_delivery_config(arguments.config)
            runtime_report = collect_runtime_report(project_root)
            if arguments.command == "parquet-delivery-plan":
                delivery_plan = plan_parquet_delivery(
                    delivery_config, runtime_report=runtime_report
                )
                payload = {
                    "schema_version": "mmprism.parquet_delivery_plan.v1",
                    "product": delivery_plan.config.product,
                    "build_id": delivery_plan.build_id,
                    "delivery_root": str(delivery_plan.delivery_root),
                    "sample_count": delivery_plan.sample_count,
                    "part_count": len(delivery_plan.parts),
                    "splits": list(delivery_plan.splits),
                    "source_manifest_sha256": delivery_plan.source_manifest_sha256,
                    "split_assignment_sha256": delivery_plan.split_assignment_sha256,
                    "estimated_payload_bytes": delivery_plan.estimated_payload_bytes,
                    "estimated_staging_bytes": delivery_plan.estimated_staging_bytes,
                    "required_free_bytes": delivery_plan.required_free_bytes,
                }
            else:
                payload = materialize_parquet_delivery(
                    delivery_config,
                    runtime_report=runtime_report,
                ).to_dict()
        elif arguments.command == "parquet-delivery-validate":
            payload = validate_parquet_delivery(
                arguments.root,
                verify_checksums=not arguments.skip_checksums,
            ).to_dict()
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
            annotation_config = load_csl_news_annotation_config(arguments.config, project_root)
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
        elif (
            arguments.command.startswith("csl-news-scheduler-")
            or arguments.command == "csl-news-annotate-scheduled"
        ):
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            annotation_config = load_csl_news_annotation_config(arguments.config, project_root)
            if arguments.command == "csl-news-scheduler-init":
                payload = initialize_csl_news_scheduler(
                    annotation_config, lease_seconds=arguments.lease_seconds
                )
            elif arguments.command == "csl-news-scheduler-pause":
                payload = set_csl_news_scheduler_state(
                    annotation_config, state="paused", reason=arguments.reason
                )
            elif arguments.command == "csl-news-scheduler-resume":
                payload = set_csl_news_scheduler_state(
                    annotation_config, state="running", reason=arguments.reason
                )
            elif arguments.command == "csl-news-scheduler-status":
                payload = build_csl_news_scheduler_status(
                    annotation_config, integrity_registry_path=arguments.integrity_registry
                )
            else:
                payload = run_csl_news_annotation_scheduled_worker(
                    annotation_config,
                    integrity_registry_path=arguments.integrity_registry,
                    worker_id=arguments.worker_id,
                    once=arguments.once,
                    poll_seconds=arguments.poll_seconds,
                )
            exit_code = 0 if payload.get("failed", 0) == 0 else 1
        elif arguments.command == "csl-news-annotation-status":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            annotation_config = load_csl_news_annotation_config(arguments.config, project_root)
            payload = build_csl_news_annotation_status(
                annotation_config,
                sample_validate_count=arguments.sample_validate,
                recent_window=arguments.recent_window,
                integrity_registry_path=arguments.integrity_registry,
            )
            if arguments.output:
                write_csl_news_annotation_status(payload, arguments.output)
            exit_code = 0 if payload["status"] == "healthy" else 1
        elif arguments.command == "csl-news-annotation-audit":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            annotation_config = load_csl_news_annotation_config(
                arguments.config, project_root
            )
            payload = build_csl_news_annotation_identity_audit(
                annotation_config,
                runtime_report=collect_runtime_report(project_root),
            )
            if arguments.output:
                write_csl_news_annotation_identity_audit(payload, arguments.output)
            exit_code = 0 if payload["status"] == "passed" else 1
        elif arguments.command == "csl-news-annotation-qc":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            annotation_config = load_csl_news_annotation_config(arguments.config, project_root)
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
            source_manifest_config = load_csl_news_source_manifest_config(arguments.config)
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
            pose_manifest_config = load_csl_news_pose_manifest_config(arguments.config)
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
        elif arguments.command == "csl-news-integrity-select":
            integrity_config = load_csl_news_integrity_config(arguments.config)
            payload = select_csl_news_integrity_archive(
                integrity_config,
                archive_id=arguments.archive_id,
                verify_sha256=not arguments.skip_sha256,
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
        elif arguments.command in {"omnihand-train", "omnihand-evaluate"}:
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            experiment_config = load_experiment_config(arguments.experiment_config)
            omnihand_run_config = load_omnihand_run_config(arguments.task_config)
            try:
                from mmprism.training.omnihand_run import (
                    evaluate_omnihand,
                    train_omnihand,
                )
            except ImportError as error:
                raise OmniHandRunError(
                    "OmniHand run dependencies are missing; install the train extra"
                ) from error
            if arguments.command == "omnihand-train":
                payload = train_omnihand(
                    experiment_config,
                    omnihand_run_config,
                    source_experiment_config=arguments.experiment_config,
                    source_task_config=arguments.task_config,
                    train_manifest_path=arguments.train_manifest,
                    validation_manifest_path=arguments.validation_manifest,
                    split_assignments_path=arguments.split_assignments,
                    resume_state_metadata_path=arguments.resume_state_metadata,
                    resume_state_tensors_path=arguments.resume_state_tensors,
                    project_root=project_root,
                    command=("mmprism", *effective_argv),
                )
            else:
                payload = evaluate_omnihand(
                    experiment_config,
                    omnihand_run_config,
                    source_experiment_config=arguments.experiment_config,
                    source_task_config=arguments.task_config,
                    manifest_path=arguments.manifest,
                    checkpoint_path=arguments.checkpoint,
                    checkpoint_metadata_path=arguments.checkpoint_metadata,
                    split_assignments_path=arguments.split_assignments,
                    split=arguments.split,
                    project_root=project_root,
                    command=("mmprism", *effective_argv),
                )
            exit_code = 0
        elif arguments.command in {"wavellm-train", "wavellm-evaluate"}:
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            experiment_config = load_experiment_config(arguments.experiment_config)
            wavellm_run_config = load_wavellm_run_config(arguments.task_config)
            model_asset_config = load_model_asset_config(arguments.model_assets)
            model_root = _resolve_mt5_model_root(arguments.model_root, project_root)
            try:
                from mmprism.training.wavellm_run import (
                    evaluate_wavellm,
                    train_wavellm,
                )
            except ImportError as error:
                raise WaveLLMRunError(
                    "WaveLLM run dependencies are missing; install the train extra"
                ) from error
            if arguments.command == "wavellm-train":
                payload = train_wavellm(
                    experiment_config,
                    wavellm_run_config,
                    model_asset_config,
                    model_root,
                    source_experiment_config=arguments.experiment_config,
                    source_task_config=arguments.task_config,
                    source_asset_config=arguments.model_assets,
                    train_manifest_path=arguments.train_manifest,
                    validation_manifest_path=arguments.validation_manifest,
                    split_assignments_path=arguments.split_assignments,
                    resume_state_metadata_path=arguments.resume_state_metadata,
                    resume_state_tensors_path=arguments.resume_state_tensors,
                    project_root=project_root,
                    command=("mmprism", *effective_argv),
                )
            else:
                payload = evaluate_wavellm(
                    experiment_config,
                    wavellm_run_config,
                    model_asset_config,
                    model_root,
                    source_experiment_config=arguments.experiment_config,
                    source_task_config=arguments.task_config,
                    source_asset_config=arguments.model_assets,
                    manifest_path=arguments.manifest,
                    checkpoint_path=arguments.checkpoint,
                    checkpoint_metadata_path=arguments.checkpoint_metadata,
                    split_assignments_path=arguments.split_assignments,
                    split=arguments.split,
                    project_root=project_root,
                    command=("mmprism", *effective_argv),
                )
            exit_code = 0
        elif arguments.command == "omnihand-smoke":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            omnihand_smoke_config = load_omnihand_smoke_config(arguments.config)
            try:
                from mmprism.training.omnihand_smoke import (
                    run_omnihand_smoke,
                    write_omnihand_smoke_report,
                )
            except ImportError as error:
                raise OmniHandSmokeError(
                    "OmniHand smoke dependencies are missing; install the train extra"
                ) from error
            payload = run_omnihand_smoke(
                omnihand_smoke_config,
                device=arguments.device,
                runtime_report=collect_runtime_report(project_root),
                command=("mmprism", *effective_argv),
            )
            write_omnihand_smoke_report(payload, arguments.output)
            exit_code = 0
        elif arguments.command == "mt5-smoke":
            project_root = (
                arguments.project_root.resolve()
                if arguments.project_root
                else discover_project_root()
            )
            mt5_config = load_mt5_smoke_config(arguments.config)
            model_config = load_model_asset_config(arguments.model_assets)
            try:
                from mmprism.training.mt5_smoke import (
                    run_mt5_smoke,
                    write_mt5_smoke_report,
                )
            except ImportError as error:
                raise MT5SmokeError(
                    "mT5 smoke dependencies are missing; install the train extra"
                ) from error
            payload = run_mt5_smoke(
                mt5_config,
                model_config,
                _resolve_mt5_model_root(arguments.model_root, project_root),
                device=arguments.device,
                runtime_report=collect_runtime_report(project_root),
                command=("mmprism", *effective_argv),
            )
            write_mt5_smoke_report(payload, arguments.output)
            exit_code = 0
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
                    model_config,
                    model_root,
                    device=arguments.device,
                    runtime_report=collect_runtime_report(project_root),
                )
                if arguments.output:
                    write_model_asset_smoke(payload, arguments.output)
                exit_code = 0
    except (
        ArtifactError,
        PrepareError,
        ConfigError,
        ManifestError,
        ModelAssetError,
        MT5SmokeError,
        OmniHandSmokeError,
        OmniHandRunError,
        WaveLLMRunError,
        ReleaseAuditError,
        CslNewsAnnotationError,
        CslNewsSchedulerError,
        CslNewsAuditError,
        CslNewsIntegrityError,
        CslNewsMetadataError,
        CslNewsPoseManifestError,
        CslNewsSourceManifestError,
        DataSplitError,
        ParquetDeliveryError,
        FileNotFoundError,
        SplitContractError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return exit_code
