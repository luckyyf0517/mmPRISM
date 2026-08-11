import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mmprism.config.schema import ExperimentConfig
from mmprism.runtime.provenance import collect_runtime_report

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RunPlan:
    run_id: str
    run_dir: Path
    created_at: str
    config_sha256: str
    resolved_config: dict[str, Any]
    runtime_report: dict[str, Any]
    expected_artifacts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "mmprism.run-plan.v1",
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "created_at": self.created_at,
            "config_sha256": self.config_sha256,
            "resolved_config": self.resolved_config,
            "runtime_report": self.runtime_report,
            "expected_artifacts": list(self.expected_artifacts),
        }


def build_run_plan(
    config: ExperimentConfig,
    project_root: Path,
    created_at: datetime | None = None,
    *,
    runtime_report: Mapping[str, Any] | None = None,
) -> RunPlan:
    resolved_config = config.resolved(project_root)
    config_payload = resolved_config.to_dict()
    config_sha256 = _stable_hash(config_payload)
    source_timestamp = created_at or datetime.now(UTC)
    if source_timestamp.tzinfo is None or source_timestamp.utcoffset() is None:
        raise ValueError("run plan timestamps must be timezone-aware")
    timestamp = source_timestamp.astimezone(UTC)
    timestamp_text = timestamp.strftime("%Y%m%dT%H%M%SZ")
    safe_name = _SAFE_NAME.sub("-", resolved_config.name).strip("-")
    run_id = f"{safe_name}__{timestamp_text}__{config_sha256[:8]}"
    run_dir = resolved_config.paths.artifact_root / resolved_config.name / run_id

    return RunPlan(
        run_id=run_id,
        run_dir=run_dir,
        created_at=timestamp.isoformat().replace("+00:00", "Z"),
        config_sha256=config_sha256,
        resolved_config=config_payload,
        runtime_report=dict(
            collect_runtime_report(project_root) if runtime_report is None else runtime_report
        ),
        expected_artifacts=(
            "run.json",
            "config.resolved.json",
            "environment.json",
            "inputs.json",
            "metrics.json",
            "predictions.jsonl",
        ),
    )
