from pathlib import Path


def discover_project_root(start: str | Path | None = None) -> Path:
    current = Path(start or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "paper" / "manager").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find the mmPRISM project root from {current}")
