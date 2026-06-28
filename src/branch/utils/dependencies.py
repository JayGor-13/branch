"""Optional dependency loading with clear research-pipeline errors."""

from __future__ import annotations

import importlib


def require(module_name: str, package_name: str | None = None):
    """Import a dependency or raise an actionable error."""

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        package = package_name or module_name
        raise ImportError(
            f"Missing dependency '{package}'. Install project dependencies with "
            "`py -m pip install -r requirements.txt` before running this step."
        ) from exc
