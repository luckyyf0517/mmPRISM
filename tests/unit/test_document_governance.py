from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_audit_module() -> ModuleType:
    script = Path(__file__).parents[2] / "scripts/audit_docs.py"
    spec = importlib.util.spec_from_file_location("mmprism_document_governance", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_audit_module()


def _authority(title: str, links: str = "") -> str:
    return f"""# {title}

Status: current
Owner: test owner
Authority scope: Test authority behavior.
Last reviewed: 2026-08-12

{links}
"""


def _minimal_repository(root: Path) -> None:
    project = root / "docs/authority"
    project.mkdir(parents=True)
    (project / "00_INDEX.md").write_text(
        _authority("Project", "[Changelog](90_CHANGELOG.md)"), encoding="utf-8"
    )
    (project / "90_CHANGELOG.md").write_text(_authority("Changelog"), encoding="utf-8")

    for name in AUDIT.WORKSPACES:
        workspace = root / "workspaces" / name
        authority = workspace / "docs/authority"
        authority.mkdir(parents=True)
        (workspace / "README.md").write_text(f"# {name}\n", encoding="utf-8")
        (authority / "00_INDEX.md").write_text(
            _authority(name, "[Changelog](90_CHANGELOG.md)"), encoding="utf-8"
        )
        (authority / "90_CHANGELOG.md").write_text(_authority("Changelog"), encoding="utf-8")


def _issue_codes(root: Path) -> set[str]:
    return {issue.code for issue in AUDIT.audit_repository(root)}


def test_valid_minimal_repository_passes(tmp_path: Path) -> None:
    _minimal_repository(tmp_path)

    assert AUDIT.audit_repository(tmp_path) == []


def test_authority_metadata_and_index_reachability_are_required(tmp_path: Path) -> None:
    _minimal_repository(tmp_path)
    scope = tmp_path / "docs/authority/10_SCOPE.md"
    scope.write_text("# Scope\n", encoding="utf-8")

    codes = _issue_codes(tmp_path)

    assert "missing-metadata" in codes
    assert "unindexed-authority" in codes


def test_broken_relative_link_is_rejected(tmp_path: Path) -> None:
    _minimal_repository(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("[Missing](docs/missing.md)\n", encoding="utf-8")

    assert "broken-link" in _issue_codes(tmp_path)


def test_compatibility_pointer_must_replace_old_authority(tmp_path: Path) -> None:
    _minimal_repository(tmp_path)
    legacy = tmp_path / "docs/architecture/README.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("# Old authority\n\nStatus: current\n", encoding="utf-8")

    codes = _issue_codes(tmp_path)

    assert "legacy-authority" in codes


def test_valid_compatibility_pointer_passes(tmp_path: Path) -> None:
    _minimal_repository(tmp_path)
    legacy = tmp_path / "docs/architecture/README.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        "# Architecture compatibility entrypoint\n\n"
        "The canonical architecture moved to:\n"
        "[project authority](../authority/00_INDEX.md)\n\n"
        "This file is retained only for links written before the migration.\n"
        "It contains no independent authority claims.\n",
        encoding="utf-8",
    )

    assert AUDIT.audit_repository(tmp_path) == []


def test_body_status_example_does_not_override_header_metadata(tmp_path: Path) -> None:
    _minimal_repository(tmp_path)
    index = tmp_path / "docs/authority/00_INDEX.md"
    index.write_text(
        _authority("Project", "[Changelog](90_CHANGELOG.md)")
        + "\nExample:\n\nStatus: completed / failed / partial\n",
        encoding="utf-8",
    )

    assert AUDIT.audit_repository(tmp_path) == []
