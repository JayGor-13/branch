"""Local environment helpers."""

from __future__ import annotations

import os
from pathlib import Path


def load_project_env(start: str | Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if one exists.

    Existing environment variables win. This intentionally supports only the
    small .env subset needed for local experiment secrets.
    """

    env_path = _find_env_file(Path(start or Path.cwd()).resolve())
    if env_path is None:
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _find_env_file(start: Path) -> Path | None:
    current = start if start.is_dir() else start.parent
    for directory in [current, *current.parents]:
        candidate = directory / ".env"
        if candidate.exists():
            return candidate
    return None
