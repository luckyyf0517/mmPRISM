#!/usr/bin/env python3
"""Validate mmPRISM Authority, workspace, and Markdown link conventions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

AUTHORITY_METADATA = ("Status", "Owner", "Authority scope", "Last reviewed")
VALID_STATUSES = {"current", "draft", "historical", "superseded"}
WORKSPACES = (
    "csl_news_annotation",
    "data_rebuild",
    "omnihand_training",
    "paper_revision",
    "wavellm_training",
)
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
METADATA_PATTERN = re.compile(r"^([A-Za-z ]+):\s*(.+?)\s*$")
COMPATIBILITY_MARKER = "compatibility entrypoint"
IGNORED_PARTS = {".git", ".venv", ".cache", "paper/manuscript"}


@dataclass(frozen=True)
class AuditIssue:
    code: str
    path: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "detail": self.detail}


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _authority_roots(root: Path) -> list[Path]:
    roots = [root / "docs/authority"]
    roots.extend(root / "workspaces" / name / "docs/authority" for name in WORKSPACES)
    return roots


def _authority_files(authority_root: Path) -> list[Path]:
    if not authority_root.is_dir():
        return []
    return sorted(authority_root.rglob("*.md"))


def _parse_metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    started = False
    for line in text.splitlines()[1:]:
        if not line.strip():
            if started:
                break
            continue
        match = METADATA_PATTERN.match(line)
        if match is None:
            if started:
                break
            continue
        started = True
        key, value = match.groups()
        metadata[key] = value.strip(" `")
    return metadata


def _link_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().split()[0].strip("<>")
    if not target or target.startswith(("#", "/", "http://", "https://", "mailto:")):
        return None
    target = unquote(target.split("#", 1)[0])
    if not target:
        return None
    return (source.parent / target).resolve()


def _markdown_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.md"):
        relative = _relative(path, root)
        if any(relative == part or relative.startswith(f"{part}/") for part in IGNORED_PARTS):
            continue
        files.append(path)
    return sorted(files)


def audit_authority(root: Path) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for authority_root in _authority_roots(root):
        relative_root = _relative(authority_root, root)
        index = authority_root / "00_INDEX.md"
        changelog = authority_root / "90_CHANGELOG.md"
        for required in (index, changelog):
            if not required.is_file():
                issues.append(
                    AuditIssue("missing-entrypoint", _relative(required, root), "required file")
                )
        files = _authority_files(authority_root)
        if not files:
            continue

        index_text = _read(index) if index.is_file() else ""
        index_targets = {
            target
            for raw_target in LINK_PATTERN.findall(index_text)
            if (target := _link_target(index, raw_target)) is not None
        }
        for path in files:
            relative = _relative(path, root)
            metadata = _parse_metadata(_read(path))
            for key in AUTHORITY_METADATA:
                if not metadata.get(key):
                    issues.append(AuditIssue("missing-metadata", relative, key))
            status = metadata.get("Status")
            if status and status not in VALID_STATUSES:
                issues.append(AuditIssue("invalid-status", relative, status))
            if path != index and path.resolve() not in index_targets:
                issues.append(
                    AuditIssue(
                        "unindexed-authority",
                        relative,
                        f"not linked from {relative_root}/00_INDEX.md",
                    )
                )
            if any(value.startswith("<") and value.endswith(">") for value in metadata.values()):
                issues.append(AuditIssue("authority-placeholder", relative, "placeholder text"))
    return issues


def audit_links(root: Path) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for path in _markdown_files(root):
        for raw_target in LINK_PATTERN.findall(_read(path)):
            target = _link_target(path, raw_target)
            if target is not None and not target.exists():
                issues.append(AuditIssue("broken-link", _relative(path, root), raw_target))
    return issues


def audit_compatibility(root: Path) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    manager_root = root / "paper/manager"
    old_roots = [root / "docs/architecture"]
    if manager_root.is_dir():
        old_roots.append(manager_root)

    for old_root in old_roots:
        for path in sorted(old_root.rglob("*.md")):
            text = _read(path)
            relative = _relative(path, root)
            if COMPATIBILITY_MARKER not in text.lower():
                issues.append(
                    AuditIssue(
                        "legacy-authority",
                        relative,
                        "old Markdown must be a compatibility entrypoint",
                    )
                )
                continue
            targets = [
                target
                for raw_target in LINK_PATTERN.findall(text)
                if (target := _link_target(path, raw_target)) is not None
            ]
            if len(targets) != 1:
                issues.append(
                    AuditIssue(
                        "invalid-compatibility",
                        relative,
                        "expected exactly one local target",
                    )
                )
            elif not targets[0].exists():
                issues.append(
                    AuditIssue("invalid-compatibility", relative, "target does not exist")
                )
            if re.search(r"^(Status|Last Updated|Role):", text, re.MULTILINE):
                issues.append(
                    AuditIssue("duplicate-current-status", relative, "legacy metadata remains")
                )
    return issues


def audit_empty_workspace_directories(root: Path) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    workspace_root = root / "workspaces"
    if not workspace_root.is_dir():
        return issues
    for path in sorted(item for item in workspace_root.rglob("*") if item.is_dir()):
        if not any(path.iterdir()):
            issues.append(
                AuditIssue("empty-directory", _relative(path, root), "remove empty directory")
            )
    return issues


def audit_repository(root: Path) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    issues.extend(audit_authority(root))
    issues.extend(audit_links(root))
    issues.extend(audit_compatibility(root))
    issues.extend(audit_empty_workspace_directories(root))
    return sorted(issues, key=lambda item: (item.path, item.code, item.detail))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="emit a machine-readable report")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    root = arguments.root.resolve()
    issues = audit_repository(root)
    if arguments.json:
        print(
            json.dumps(
                {
                    "root": str(root),
                    "valid": not issues,
                    "issues": [item.as_dict() for item in issues],
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
        )
    elif issues:
        for issue in issues:
            print(f"{issue.path}: {issue.code}: {issue.detail}", file=sys.stderr)
    else:
        print("Documentation governance audit passed.")
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
