from __future__ import annotations

import subprocess
from pathlib import Path
from urllib.parse import urlparse

from app.logging_config import get_logger

logger = get_logger(__name__)


def load_repository(repo: str, destination_root: Path) -> Path:
    """Clone a remote repository or return an existing local repository path."""
    source = Path(repo).expanduser()
    if source.exists():
        logger.info("Using local repository: %s", source.resolve())
        return source.resolve()

    destination_root.mkdir(parents=True, exist_ok=True)
    repo_name = _repo_name_from_url(repo)
    destination = destination_root / repo_name

    if destination.exists():
        logger.info("Repository exists at %s; pulling latest changes", destination)
        _run_git(["git", "-C", str(destination), "pull", "--ff-only"])
    else:
        logger.info("Cloning repository %s -> %s", repo, destination)
        _run_git(["git", "clone", repo, str(destination)])

    return destination.resolve()


def current_commit(repo_path: Path) -> str | None:
    try:
        return _run_git(["git", "-C", str(repo_path), "rev-parse", "HEAD"]).strip()
    except RuntimeError:
        logger.debug("Could not determine commit for %s", repo_path)
        return None


def _repo_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name.removesuffix(".git") or "repository"


def _run_git(args: list[str]) -> str:
    logger.debug("Running git: %s", " ".join(args))
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout

