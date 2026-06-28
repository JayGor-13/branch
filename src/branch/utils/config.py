"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from branch.utils.dependencies import require


def load_yaml(path: str | Path) -> dict[str, Any]:
    yaml = require("yaml", "PyYAML")
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
