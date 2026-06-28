"""Filesystem and JSON helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(payload: Any, path: str | Path) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    return out


def write_text(text: str, path: str | Path) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(text, encoding="utf-8")
    return out


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]
