from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

RELEASE_AUDIT_CONFIG_SCHEMA = "mmprism.release_audit_config.v1"
RELEASE_AUDIT_REPORT_SCHEMA = "mmprism.release_audit_report.v1"
_GLOB_META = re.compile(r"[*?[]")


class ReleaseAuditError(ValueError):
    """Raised when a release audit cannot be configured or executed safely."""


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReleaseAuditError(f"{location} must be a mapping")
    return value


def _reject_unknown(payload: Mapping[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ReleaseAuditError(f"Unknown keys in {location}: {', '.join(unknown)}")


def _text(payload: Mapping[str, Any], key: str, location: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReleaseAuditError(f"{location}.{key} must be a non-empty string")
    return value.strip()


def _relative_pattern(value: str, location: str) -> str:
    if "\\" in value:
        raise ReleaseAuditError(f"{location} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ReleaseAuditError(f"{location} must stay within the project root: {value}")
    return value


def _string_tuple(
    payload: Mapping[str, Any], key: str, location: str, *, required: bool = False
) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None and not required:
        return ()
    if not isinstance(value, list) or not value:
        raise ReleaseAuditError(f"{location}.{key} must be a non-empty list")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ReleaseAuditError(f"{location}.{key}[{index}] must be non-empty text")
        result.append(_relative_pattern(item.strip(), f"{location}.{key}[{index}]"))
    if len(result) != len(set(result)):
        raise ReleaseAuditError(f"{location}.{key} must not contain duplicates")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ReleaseAuditConfig:
    release_id: str
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    required_paths: tuple[str, ...]
    forbidden_paths: tuple[str, ...]
    forbidden_content_patterns: tuple[tuple[str, str], ...]
    python_roots: tuple[str, ...]
    forbidden_import_prefixes: tuple[str, ...]
    expected_entrypoints: tuple[tuple[str, str], ...]
    require_clean_git: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RELEASE_AUDIT_CONFIG_SCHEMA,
            "release_id": self.release_id,
            "include": list(self.include),
            "exclude": list(self.exclude),
            "required_paths": list(self.required_paths),
            "forbidden_paths": list(self.forbidden_paths),
            "forbidden_content_patterns": dict(self.forbidden_content_patterns),
            "python_roots": list(self.python_roots),
            "forbidden_import_prefixes": list(self.forbidden_import_prefixes),
            "expected_entrypoints": dict(self.expected_entrypoints),
            "require_clean_git": self.require_clean_git,
        }

    @property
    def fingerprint(self) -> str:
        serialized = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    @classmethod
    def from_mapping(cls, value: object) -> ReleaseAuditConfig:
        payload = _mapping(value, "release audit config")
        allowed = {
            "schema_version",
            "release_id",
            "include",
            "exclude",
            "required_paths",
            "forbidden_paths",
            "forbidden_content_patterns",
            "python_roots",
            "forbidden_import_prefixes",
            "expected_entrypoints",
            "require_clean_git",
        }
        _reject_unknown(payload, allowed, "release audit config")
        if payload.get("schema_version") != RELEASE_AUDIT_CONFIG_SCHEMA:
            raise ReleaseAuditError(
                f"Unsupported release audit schema: {payload.get('schema_version')!r}"
            )

        pattern_payload = _mapping(
            payload.get("forbidden_content_patterns", {}),
            "release audit config.forbidden_content_patterns",
        )
        content_patterns: list[tuple[str, str]] = []
        for name, pattern in sorted(pattern_payload.items()):
            if not isinstance(name, str) or not name.strip():
                raise ReleaseAuditError("forbidden content pattern names must be non-empty")
            if not isinstance(pattern, str) or not pattern:
                raise ReleaseAuditError(f"forbidden content pattern {name!r} must be text")
            try:
                re.compile(pattern)
            except re.error as error:
                raise ReleaseAuditError(
                    f"invalid forbidden content pattern {name!r}: {error}"
                ) from error
            content_patterns.append((name, pattern))

        entrypoint_payload = _mapping(
            payload.get("expected_entrypoints", {}),
            "release audit config.expected_entrypoints",
        )
        entrypoints: list[tuple[str, str]] = []
        for name, target in sorted(entrypoint_payload.items()):
            if not isinstance(name, str) or not name.strip():
                raise ReleaseAuditError("entrypoint names must be non-empty")
            if not isinstance(target, str) or not re.fullmatch(
                r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", target
            ):
                raise ReleaseAuditError(f"invalid entrypoint target for {name!r}: {target!r}")
            entrypoints.append((name, target))

        clean = payload.get("require_clean_git", True)
        if not isinstance(clean, bool):
            raise ReleaseAuditError("release audit config.require_clean_git must be boolean")

        import_prefixes = _string_tuple(
            payload, "forbidden_import_prefixes", "release audit config"
        )
        for index, prefix in enumerate(import_prefixes):
            if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", prefix):
                raise ReleaseAuditError(
                    "release audit config.forbidden_import_prefixes"
                    f"[{index}] is not a Python module prefix"
                )

        required_paths = _string_tuple(
            payload, "required_paths", "release audit config", required=True
        )
        if any(_GLOB_META.search(path) for path in required_paths):
            raise ReleaseAuditError("required_paths must be literal relative paths, not globs")

        return cls(
            release_id=_text(payload, "release_id", "release audit config"),
            include=_string_tuple(
                payload, "include", "release audit config", required=True
            ),
            exclude=_string_tuple(payload, "exclude", "release audit config"),
            required_paths=required_paths,
            forbidden_paths=_string_tuple(
                payload, "forbidden_paths", "release audit config"
            ),
            forbidden_content_patterns=tuple(content_patterns),
            python_roots=_string_tuple(
                payload, "python_roots", "release audit config", required=True
            ),
            forbidden_import_prefixes=import_prefixes,
            expected_entrypoints=tuple(entrypoints),
            require_clean_git=clean,
        )


@dataclass(frozen=True, slots=True)
class RepositorySnapshot:
    commit: str
    state: str
    tracked_paths: tuple[str, ...]
    dirty_entries: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Module:
    name: str
    path: str
    is_package: bool


def load_release_audit_config(path: str | Path) -> ReleaseAuditConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ReleaseAuditError(f"Release audit config does not exist: {config_path}")
    try:
        payload: object = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ReleaseAuditError(f"Unable to load release audit config: {error}") from error
    return ReleaseAuditConfig.from_mapping(payload)


def _run_git(project_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ("git", "-C", str(project_root), *arguments),
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReleaseAuditError(f"Unable to inspect Git repository: {error}") from error
    return completed.stdout


def capture_repository_snapshot(project_root: str | Path) -> RepositorySnapshot:
    root = Path(project_root).expanduser().resolve()
    commit = _run_git(root, "rev-parse", "HEAD").decode("ascii").strip()
    tracked = tuple(
        sorted(
            item.decode("utf-8")
            for item in _run_git(root, "ls-files", "-z").split(b"\0")
            if item
        )
    )
    dirty = tuple(
        line
        for line in _run_git(
            root, "status", "--porcelain=v1", "--untracked-files=all"
        )
        .decode("utf-8")
        .splitlines()
        if line
    )
    return RepositorySnapshot(
        commit=commit,
        state="clean" if not dirty else "dirty",
        tracked_paths=tracked,
        dirty_entries=dirty,
    )


def _matches(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatchcase(path, pattern) for pattern in patterns)


def _finding(
    code: str,
    message: str,
    *,
    path: str | None = None,
    line: int | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {"code": code, "message": message}
    if path is not None:
        result["path"] = path
    if line is not None:
        result["line"] = line
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _discover_modules(project_root: Path, roots: Sequence[str]) -> tuple[_Module, ...]:
    modules: list[_Module] = []
    for root_text in roots:
        root = project_root / root_text
        if not root.is_dir():
            continue
        import_base = root.parent
        for path in sorted(root.rglob("*.py")):
            relative = path.relative_to(import_base)
            parts = list(relative.with_suffix("").parts)
            is_package = parts[-1] == "__init__"
            if is_package:
                parts.pop()
            modules.append(
                _Module(
                    name=".".join(parts),
                    path=path.relative_to(project_root).as_posix(),
                    is_package=is_package,
                )
            )
    return tuple(modules)


def _has_prefix(module: str, prefixes: Sequence[str]) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)


def _strongly_connected_components(
    graph: Mapping[str, set[str]],
) -> tuple[tuple[str, ...], ...]:
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    low_links: dict[str, int] = {}
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        low_links[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in sorted(graph.get(node, set())):
            if target not in indices:
                visit(target)
                low_links[node] = min(low_links[node], low_links[target])
            elif target in on_stack:
                low_links[node] = min(low_links[node], indices[target])
        if low_links[node] != indices[node]:
            return
        component: list[str] = []
        while stack:
            target = stack.pop()
            on_stack.remove(target)
            component.append(target)
            if target == node:
                break
        ordered = tuple(sorted(component))
        if len(ordered) > 1 or node in graph.get(node, set()):
            components.append(ordered)

    for module in sorted(graph):
        if module not in indices:
            visit(module)
    return tuple(sorted(components))


def _dependency_audit(
    project_root: Path,
    config: ReleaseAuditConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    modules = _discover_modules(project_root, config.python_roots)
    module_names = {module.name for module in modules}
    graph: dict[str, set[str]] = {module.name: set() for module in modules}
    edges: set[tuple[str, str]] = set()
    external: set[str] = set()
    findings: list[dict[str, object]] = []

    for module in modules:
        path = project_root / module.path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=module.path)
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            findings.append(
                _finding("PYTHON_PARSE_ERROR", str(error), path=module.path)
            )
            continue
        for node in ast.walk(tree):
            imported: list[str] = []
            line_number = getattr(node, "lineno", None)
            line = line_number if isinstance(line_number, int) else None
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    findings.append(
                        _finding(
                            "RELATIVE_IMPORT",
                            "canonical modules must use absolute mmprism.* imports",
                            path=module.path,
                            line=line,
                        )
                    )
                    continue
                if node.module is not None:
                    imported.append(node.module)
            for target in imported:
                if _has_prefix(target, config.forbidden_import_prefixes):
                    findings.append(
                        _finding(
                            "FORBIDDEN_IMPORT",
                            f"canonical module imports legacy namespace {target!r}",
                            path=module.path,
                            line=line,
                        )
                    )
                if target == "mmprism" or target.startswith("mmprism."):
                    if target not in module_names:
                        findings.append(
                            _finding(
                                "MISSING_INTERNAL_MODULE",
                                f"import target {target!r} does not exist",
                                path=module.path,
                                line=line,
                            )
                        )
                    else:
                        graph[module.name].add(target)
                        edges.add((module.name, target))
                else:
                    top_level = target.partition(".")[0]
                    if top_level not in sys.stdlib_module_names:
                        external.add(top_level)

    cycles = _strongly_connected_components(graph)
    for cycle in cycles:
        findings.append(
            _finding(
                "IMPORT_CYCLE",
                f"canonical import cycle detected: {' -> '.join(cycle)}",
            )
        )
    return (
        {
            "module_count": len(modules),
            "modules": [module.name for module in modules],
            "internal_edge_count": len(edges),
            "internal_edges": [list(edge) for edge in sorted(edges)],
            "external_imports": sorted(external),
            "cycles": [list(cycle) for cycle in cycles],
        },
        findings,
    )


def _entrypoint_audit(
    project_root: Path,
    config: ReleaseAuditConfig,
    module_names: set[str],
) -> tuple[dict[str, str], list[dict[str, object]]]:
    path = project_root / "pyproject.toml"
    findings: list[dict[str, object]] = []
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        scripts_value = payload.get("project", {}).get("scripts", {})
        scripts = {
            str(name): str(target)
            for name, target in _mapping(scripts_value, "project.scripts").items()
        }
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, ReleaseAuditError) as error:
        return {}, [_finding("PYPROJECT_ERROR", str(error), path="pyproject.toml")]

    for name, expected in config.expected_entrypoints:
        actual = scripts.get(name)
        if actual != expected:
            findings.append(
                _finding(
                    "ENTRYPOINT_MISMATCH",
                    f"entrypoint {name!r}: expected {expected!r}, got {actual!r}",
                    path="pyproject.toml",
                )
            )
            continue
        module_name = expected.partition(":")[0]
        if module_name not in module_names:
            findings.append(
                _finding(
                    "ENTRYPOINT_MODULE_MISSING",
                    f"entrypoint module does not exist: {module_name}",
                    path="pyproject.toml",
                )
            )
    return dict(sorted(scripts.items())), findings


def audit_release(
    config: ReleaseAuditConfig,
    *,
    project_root: str | Path,
    repository_snapshot: RepositorySnapshot | None = None,
) -> dict[str, object]:
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise ReleaseAuditError(f"Project root does not exist: {root}")
    snapshot = repository_snapshot or capture_repository_snapshot(root)
    tracked = tuple(sorted(set(snapshot.tracked_paths)))
    findings: list[dict[str, object]] = []

    if config.require_clean_git and snapshot.state != "clean":
        findings.append(
            _finding(
                "GIT_DIRTY",
                f"release audit requires clean Git state ({len(snapshot.dirty_entries)} entries)",
            )
        )

    included: set[str] = set()
    for pattern in config.include:
        matches = {path for path in tracked if fnmatchcase(path, pattern)}
        included.update(matches)
        if _GLOB_META.search(pattern) and not matches:
            findings.append(
                _finding(
                    "INCLUDE_PATTERN_UNMATCHED",
                    f"include pattern matched no tracked files: {pattern}",
                )
            )
    selected = tuple(
        sorted(path for path in included if not _matches(path, config.exclude))
    )

    for required in config.required_paths:
        path = root / required
        if not path.is_file():
            findings.append(
                _finding(
                    "REQUIRED_PATH_MISSING",
                    "required release file does not exist",
                    path=required,
                )
            )
        elif required not in selected:
            findings.append(
                _finding(
                    "REQUIRED_PATH_NOT_SELECTED",
                    "required release file is not selected by include rules",
                    path=required,
                )
            )

    selected_files: list[dict[str, object]] = []
    content_regexes = [
        (name, re.compile(pattern)) for name, pattern in config.forbidden_content_patterns
    ]
    for relative in selected:
        path = root / relative
        if _matches(relative, config.forbidden_paths):
            findings.append(
                _finding(
                    "FORBIDDEN_PATH_SELECTED",
                    "internal or legacy path was selected for release",
                    path=relative,
                )
            )
        if not path.is_file():
            findings.append(
                _finding(
                    "SELECTED_PATH_MISSING",
                    "tracked release path is not a regular file",
                    path=relative,
                )
            )
            continue
        if path.is_symlink():
            findings.append(
                _finding(
                    "SYMLINK_SELECTED",
                    "release files must not rely on symlinks",
                    path=relative,
                )
            )
            continue
        selected_files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, regex in content_regexes:
            for match in regex.finditer(text):
                findings.append(
                    _finding(
                        "FORBIDDEN_CONTENT",
                        f"matched forbidden content rule {name!r}",
                        path=relative,
                        line=text.count("\n", 0, match.start()) + 1,
                    )
                )

    dependency, dependency_findings = _dependency_audit(root, config)
    findings.extend(dependency_findings)
    module_values = dependency.get("modules")
    if not isinstance(module_values, list) or not all(
        isinstance(item, str) for item in module_values
    ):
        raise ReleaseAuditError("Internal dependency audit produced invalid module metadata")
    module_names = set(module_values)
    entrypoints, entrypoint_findings = _entrypoint_audit(root, config, module_names)
    findings.extend(entrypoint_findings)

    def finding_key(item: Mapping[str, object]) -> tuple[str, str, int, str]:
        line_value = item.get("line")
        return (
            str(item["code"]),
            str(item.get("path", "")),
            line_value if isinstance(line_value, int) else 0,
            str(item["message"]),
        )

    findings.sort(
        key=finding_key
    )

    forbidden_tracked = sorted(
        path for path in tracked if _matches(path, config.forbidden_paths)
    )
    return {
        "schema_version": RELEASE_AUDIT_REPORT_SCHEMA,
        "release_id": config.release_id,
        "config_fingerprint": config.fingerprint,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "passed" if not findings else "failed",
        "git": {
            "commit": snapshot.commit,
            "state": snapshot.state,
            "dirty_entry_count": len(snapshot.dirty_entries),
        },
        "selection": {
            "tracked_file_count": len(tracked),
            "selected_file_count": len(selected_files),
            "selected_bytes": sum(
                value
                for item in selected_files
                if isinstance((value := item.get("bytes")), int)
            ),
            "forbidden_tracked_but_excluded_count": len(forbidden_tracked),
            "forbidden_tracked_but_excluded": forbidden_tracked,
            "files": selected_files,
        },
        "entrypoints": entrypoints,
        "dependencies": dependency,
        "finding_count": len(findings),
        "findings": findings,
    }


def write_release_audit(payload: Mapping[str, object], path: str | Path) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
