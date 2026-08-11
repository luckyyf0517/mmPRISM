from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from mmprism.artifacts.run import RunInput, RunInputKind
from mmprism.config import ExperimentConfig, load_experiment_config
from mmprism.contracts import SplitIndex, validate_manifest, validate_split_assignments
from mmprism.runtime import build_run_plan, collect_runtime_report

PREPARE_REPORT_SCHEMA_VERSION = "mmprism.prepare_report.v1"
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class PrepareError(ValueError):
    """Raised when a formal run cannot pass its side-effect-free preflight."""


def _validate_formal_runtime(report: Mapping[str, Any], project_root: Path) -> None:
    reported_root = report.get("project_root")
    if not isinstance(reported_root, str) or Path(reported_root).resolve() != project_root:
        raise PrepareError("prepare requires matching project-root provenance")
    git = report.get("git")
    if (
        not isinstance(git, Mapping)
        or not isinstance(git.get("commit"), str)
        or not _GIT_COMMIT.fullmatch(git["commit"])
    ):
        raise PrepareError("prepare requires a valid Git commit")
    if git.get("dirty") is not False:
        raise PrepareError("prepare requires a clean Git worktree")


def _writable_destination(path: Path, name: str) -> dict[str, object]:
    candidate = path.expanduser().resolve()
    if candidate.exists():
        if not candidate.is_dir():
            raise PrepareError(f"{name} exists but is not a directory: {candidate}")
        ancestor = candidate
    else:
        ancestor = candidate
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            raise PrepareError(f"{name} has no existing parent directory: {candidate}")
    if not os.access(ancestor, os.W_OK | os.X_OK):
        raise PrepareError(f"{name} is not writable through {ancestor}")
    return {
        "path": str(candidate),
        "exists": candidate.exists(),
        "writable_ancestor": str(ancestor),
    }


def validate_split_bindings(
    manifest_sample_ids: Mapping[str, Iterable[str]],
    split_assignments_path: str | Path,
    bindings: Mapping[str, str],
) -> dict[str, object]:
    """Validate that disjoint manifest samples have their declared split assignment."""

    manifest_names = set(manifest_sample_ids)
    if not manifest_names:
        raise PrepareError("split binding requires at least one manifest")
    if set(bindings) != manifest_names:
        missing = sorted(manifest_names - set(bindings))
        extra = sorted(set(bindings) - manifest_names)
        raise PrepareError(
            "split bindings must cover every manifest exactly once: "
            f"missing={missing}, extra={extra}"
        )
    if any(not isinstance(split, str) or not split.strip() for split in bindings.values()):
        raise PrepareError("split binding values must be non-empty split names")

    path = Path(split_assignments_path).expanduser().resolve()
    summary = validate_split_assignments(path)
    index = SplitIndex(path)
    known_splits = set(index.splits)
    unknown_splits = sorted(set(bindings.values()) - known_splits)
    if unknown_splits:
        raise PrepareError(
            f"split bindings reference unknown assignments: {unknown_splits}"
        )

    seen_samples: dict[str, str] = {}
    binding_reports: dict[str, object] = {}
    bound_sample_ids: set[str] = set()
    for manifest_name in sorted(manifest_names):
        expected_split = bindings[manifest_name]
        sample_ids = set(manifest_sample_ids[manifest_name])
        if not sample_ids:
            raise PrepareError(f"manifest {manifest_name!r} has no sample IDs")
        overlap = sorted(sample_ids & set(seen_samples))
        if overlap:
            raise PrepareError(
                f"manifest inputs overlap on {len(overlap)} sample IDs: {overlap[:5]}"
            )
        seen_samples.update({sample_id: manifest_name for sample_id in sample_ids})
        missing = sorted(sample_id for sample_id in sample_ids if sample_id not in index)
        wrong = sorted(
            sample_id
            for sample_id in sample_ids
            if sample_id in index and index[sample_id].split != expected_split
        )
        if missing or wrong:
            raise PrepareError(
                f"manifest {manifest_name!r} does not match split {expected_split!r}: "
                f"missing={len(missing)} {missing[:5]}, wrong_split={len(wrong)} {wrong[:5]}"
            )
        bound_sample_ids.update(sample_ids)
        binding_reports[manifest_name] = {
            "split": expected_split,
            "sample_count": len(sample_ids),
        }

    return {
        "assignments": summary.to_dict(),
        "bindings": binding_reports,
        "bound_sample_count": len(bound_sample_ids),
        "unbound_assignment_count": summary.assignment_count - len(bound_sample_ids),
    }


def build_prepare_report(
    experiment_config: ExperimentConfig,
    *,
    source_config: str | Path,
    inputs: Sequence[RunInput],
    split_bindings: Mapping[str, str],
    project_root: str | Path,
    runtime_report: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Build a complete run preflight without creating directories or artifacts."""

    root = Path(project_root).expanduser().resolve()
    report = dict(collect_runtime_report(root) if runtime_report is None else runtime_report)
    _validate_formal_runtime(report, root)
    source_path = Path(source_config).expanduser()
    source_path = (
        source_path.resolve()
        if source_path.is_absolute()
        else (root / source_path).resolve()
    )
    if not source_path.is_file():
        raise PrepareError(f"source configuration is not a file: {source_path}")
    reloaded = load_experiment_config(source_path)
    resolved = experiment_config.resolved(root)
    if reloaded.resolved(root).to_dict() != resolved.to_dict():
        raise PrepareError("source configuration does not match the loaded experiment config")
    if not resolved.paths.data_root.is_dir():
        raise PrepareError(f"data root does not exist: {resolved.paths.data_root}")

    plan = build_run_plan(
        experiment_config,
        root,
        created_at=created_at,
        runtime_report=report,
    )
    if plan.run_dir.exists():
        raise PrepareError(f"planned run directory already exists: {plan.run_dir}")

    if not inputs:
        raise PrepareError("prepare requires registered inputs")
    input_names = [item.name for item in inputs]
    if len(set(input_names)) != len(input_names):
        raise PrepareError("prepare input names must be unique")
    manifest_inputs = [item for item in inputs if item.kind is RunInputKind.MANIFEST]
    split_inputs = [item for item in inputs if item.kind is RunInputKind.SPLIT]
    if not manifest_inputs:
        raise PrepareError("prepare requires at least one manifest input")
    if len(split_inputs) != 1:
        raise PrepareError("prepare requires exactly one split input")

    manifest_summaries: dict[str, object] = {}
    manifest_ids: dict[str, frozenset[str]] = {}
    input_reports: list[dict[str, object]] = []
    for item in sorted(inputs, key=lambda value: value.name):
        validation: dict[str, object] = {"kind": item.kind.value, "status": "captured"}
        if item.kind is RunInputKind.MANIFEST:
            summary = validate_manifest(item.path)
            manifest_ids[item.name] = summary.sample_ids
            validation = {"kind": item.kind.value, "status": "passed", **summary.to_dict()}
            manifest_summaries[item.name] = summary.to_dict()
        elif item.kind is RunInputKind.SPLIT:
            validation = {
                "kind": item.kind.value,
                "status": "passed",
                **validate_split_assignments(item.path).to_dict(),
            }
        RunInput.capture(
            name=item.name,
            kind=item.kind,
            path=item.path,
            expected_sha256=item.sha256,
        )
        input_reports.append({**item.to_dict(), "validation": validation})

    binding_report = validate_split_bindings(
        manifest_ids,
        split_inputs[0].path,
        split_bindings,
    )
    source_input = RunInput.capture(name="experiment_config", kind="config", path=source_path)
    destinations = {
        "artifact_root": _writable_destination(resolved.paths.artifact_root, "artifact root"),
        "cache_root": _writable_destination(resolved.paths.cache_root, "cache root"),
    }
    return {
        "schema_version": PREPARE_REPORT_SCHEMA_VERSION,
        "status": "passed",
        "side_effect_free": True,
        "source_config": source_input.to_dict(),
        "plan": plan.to_dict(),
        "inputs": input_reports,
        "validation": {
            "clean_git": True,
            "data_root": str(resolved.paths.data_root),
            "planned_run_dir_absent": True,
            "manifest_count": len(manifest_inputs),
            "manifest_record_count": sum(
                int(summary["record_count"])
                for summary in manifest_summaries.values()
                if isinstance(summary, Mapping)
            ),
            "split": binding_report,
            "destinations": destinations,
        },
    }
