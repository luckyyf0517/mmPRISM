from __future__ import annotations

import json
from pathlib import Path

import pytest

from mmprism.cli import main
from mmprism.release import (
    ReleaseAuditConfig,
    ReleaseAuditError,
    RepositorySnapshot,
    audit_release,
    load_release_audit_config,
    write_release_audit,
)


def _write_project(root: Path, *, core_source: str = "import json\n") -> tuple[str, ...]:
    files = {
        "README.md": "# Fixture\n",
        "pyproject.toml": (
            "[project]\nname = 'fixture'\nversion = '0.0.0'\n"
            "[project.scripts]\nmmprism = 'mmprism.cli:main'\n"
        ),
        "src/mmprism/__init__.py": "\n",
        "src/mmprism/cli.py": "from mmprism.core import run\n",
        "src/mmprism/core.py": f"{core_source}\ndef run():\n    return None\n",
        "CLAUDE.md": "internal\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tuple(sorted(files))


def _config(
    *,
    required_paths: list[str] | None = None,
    forbidden_content_patterns: dict[str, str] | None = None,
) -> ReleaseAuditConfig:
    return ReleaseAuditConfig.from_mapping(
        {
            "schema_version": "mmprism.release_audit_config.v1",
            "release_id": "fixture-v1",
            "include": ["README.md", "pyproject.toml", "src/mmprism/**"],
            "exclude": ["**/__pycache__/**"],
            "required_paths": required_paths
            or ["README.md", "pyproject.toml", "src/mmprism/cli.py"],
            "forbidden_paths": ["CLAUDE.md", "src/fmcw/**"],
            "forbidden_content_patterns": forbidden_content_patterns
            or {"local_path": r"/(?:home|root)/[^\s]+"},
            "python_roots": ["src/mmprism"],
            "forbidden_import_prefixes": ["src.fmcw"],
            "expected_entrypoints": {"mmprism": "mmprism.cli:main"},
            "require_clean_git": True,
        }
    )


def _snapshot(paths: tuple[str, ...], *, state: str = "clean") -> RepositorySnapshot:
    return RepositorySnapshot(
        commit="a" * 40,
        state=state,
        tracked_paths=paths,
        dirty_entries=(" M README.md",) if state == "dirty" else (),
    )


def test_release_audit_hashes_selected_files_and_excludes_internal_paths(
    tmp_path: Path,
) -> None:
    tracked = _write_project(tmp_path)

    report = audit_release(
        _config(), project_root=tmp_path, repository_snapshot=_snapshot(tracked)
    )

    assert report["status"] == "passed"
    assert report["config_fingerprint"] == _config().fingerprint
    assert report["finding_count"] == 0
    selection = report["selection"]
    assert selection["selected_file_count"] == 5
    assert selection["forbidden_tracked_but_excluded"] == ["CLAUDE.md"]
    assert report["dependencies"]["cycles"] == []
    assert report["entrypoints"] == {"mmprism": "mmprism.cli:main"}


def test_release_audit_reports_missing_required_legacy_import_and_dirty_git(
    tmp_path: Path,
) -> None:
    tracked = _write_project(tmp_path, core_source="import src.fmcw.simulator\n")

    report = audit_release(
        _config(required_paths=["README.md", "LICENSE"]),
        project_root=tmp_path,
        repository_snapshot=_snapshot(tracked, state="dirty"),
    )

    assert report["status"] == "failed"
    codes = {finding["code"] for finding in report["findings"]}
    assert codes == {"FORBIDDEN_IMPORT", "GIT_DIRTY", "REQUIRED_PATH_MISSING"}


def test_release_audit_detects_internal_import_cycle(tmp_path: Path) -> None:
    tracked = list(_write_project(tmp_path))
    (tmp_path / "src/mmprism/a.py").write_text("import mmprism.b\n", encoding="utf-8")
    (tmp_path / "src/mmprism/b.py").write_text("import mmprism.a\n", encoding="utf-8")
    tracked.extend(("src/mmprism/a.py", "src/mmprism/b.py"))

    report = audit_release(
        _config(),
        project_root=tmp_path,
        repository_snapshot=_snapshot(tuple(sorted(tracked))),
    )

    assert report["status"] == "failed"
    assert report["dependencies"]["cycles"] == [["mmprism.a", "mmprism.b"]]
    assert any(item["code"] == "IMPORT_CYCLE" for item in report["findings"])


def test_release_audit_rejects_an_unsupported_model_claim(tmp_path: Path) -> None:
    tracked = _write_project(tmp_path)
    (tmp_path / "README.md").write_text(
        "# Fixture\nSupported backend: " + "Phi" + "-3\n", encoding="utf-8"
    )

    report = audit_release(
        _config(
            forbidden_content_patterns={
                "unsupported_language_backend": r"(?i)ph[i][-_ ]?3"
            }
        ),
        project_root=tmp_path,
        repository_snapshot=_snapshot(tracked),
    )

    assert report["status"] == "failed"
    assert any(
        finding["code"] == "FORBIDDEN_CONTENT"
        and finding["path"] == "README.md"
        and finding["line"] == 2
        for finding in report["findings"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"surprise": True},
        {"include": ["../outside"]},
        {"required_paths": ["*.md"]},
        {"forbidden_content_patterns": {"broken": "["}},
    ],
)
def test_release_config_rejects_unknown_or_unsafe_values(
    mutation: dict[str, object],
) -> None:
    payload: dict[str, object] = {
        "schema_version": "mmprism.release_audit_config.v1",
        "release_id": "fixture-v1",
        "include": ["README.md"],
        "required_paths": ["README.md"],
        "python_roots": ["src/mmprism"],
    }
    payload.update(mutation)
    with pytest.raises(ReleaseAuditError):
        ReleaseAuditConfig.from_mapping(payload)


def test_load_and_atomic_write_release_audit(tmp_path: Path) -> None:
    config_path = tmp_path / "release.yaml"
    config_path.write_text(
        """\
schema_version: mmprism.release_audit_config.v1
release_id: fixture-v1
include: [README.md]
required_paths: [README.md]
python_roots: [src/mmprism]
""",
        encoding="utf-8",
    )
    config = load_release_audit_config(config_path)
    assert config.release_id == "fixture-v1"

    output = tmp_path / "reports" / "release.json"
    write_release_audit({"status": "passed"}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "passed"}


def test_release_audit_cli_writes_a_machine_readable_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tracked = _write_project(tmp_path)
    config_path = tmp_path / "release.yaml"
    config_path.write_text(
        """\
schema_version: mmprism.release_audit_config.v1
release_id: fixture-v1
include: [README.md, pyproject.toml, src/mmprism/**]
required_paths: [README.md, pyproject.toml, src/mmprism/cli.py]
python_roots: [src/mmprism]
expected_entrypoints:
  mmprism: mmprism.cli:main
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "mmprism.release.audit.capture_repository_snapshot", lambda _: _snapshot(tracked)
    )
    output = tmp_path / "report.json"

    exit_code = main(
        [
            "release-audit",
            str(config_path),
            "--project-root",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "passed"
    assert json.loads(capsys.readouterr().out)["schema_version"].endswith(".v1")
