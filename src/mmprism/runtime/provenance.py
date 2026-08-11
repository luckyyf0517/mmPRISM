import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git_output(project_root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime_report(project_root: Path) -> dict[str, Any]:
    commit = _git_output(project_root, "rev-parse", "HEAD")
    status = _git_output(project_root, "status", "--porcelain")
    packages = {
        name: _package_version(name)
        for name in (
            "numpy",
            "PyYAML",
            "torch",
            "safetensors",
            "lightning",
            "transformers",
            "peft",
        )
    }
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "project_root": str(project_root),
        "git": {
            "commit": commit,
            "dirty": bool(status) if status is not None else None,
        },
        "packages": packages,
    }
