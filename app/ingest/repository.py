from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import urlparse


def load_repository(repo: str, destination_root: Path) -> Path:
    """Clone a remote repository or return an existing local repository path."""
    source = Path(repo).expanduser()
    if source.exists():
        return source.resolve()

    destination_root.mkdir(parents=True, exist_ok=True)
    repo_name = _repo_name_from_url(repo)
    destination = destination_root / repo_name

    if destination.exists():
        _run_git(["git", "-C", str(destination), "pull", "--ff-only"])
    else:
        _run_git(["git", "clone", repo, str(destination)])

    return destination.resolve()


def current_commit(repo_path: Path) -> str | None:
    try:
        return _run_git(["git", "-C", str(repo_path), "rev-parse", "HEAD"]).strip()
    except RuntimeError:
        return None


def _repo_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name.removesuffix(".git") or "repository"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout

