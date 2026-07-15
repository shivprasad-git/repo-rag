from __future__ import annotations

from pathlib import Path

from app.config import Settings


def discover_files(repo_path: Path, settings: Settings) -> list[Path]:
    files: list[Path] = []
    ignored = set(settings.ignored_dirs)

    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored for part in path.parts):
            continue
        if path.suffix.lower() in settings.supported_extensions:
            files.append(path)

    return sorted(files)

