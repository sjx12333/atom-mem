"""Deployment-local paths for the Graphiti memory module.

Every persisted Graphiti asset is rooted below this checkout.  Callers may
pass relative paths, but resolution is anchored to the module directory rather
than the process working directory, so starting the server from another
directory does not leak data into the user's profile or an external volume.
"""

from __future__ import annotations

import os
from pathlib import Path


#: Code root: ``app/memos/memos_system`` inside the standalone package.
MEMOS_SYSTEM_ROOT = Path(__file__).resolve().parents[1]
#: Data/config root.  The launcher sets ``ATOM_MEM_HOME`` to the package root
#: (the directory containing ``4.MEMOS-API.bat``), so runtime state never lives
#: inside the code tree and survives an application update.
HOME_ROOT = Path(os.environ.get("ATOM_MEM_HOME") or MEMOS_SYSTEM_ROOT).resolve()
DATA_ROOT = HOME_ROOT / "data"
GRAPHITI_ROOT = DATA_ROOT / "graphiti"
CONFIG_ROOT = HOME_ROOT / "config"


def resolve_local_path(value: str | Path, *, base: Path = HOME_ROOT) -> Path:
    """Resolve a path inside the package root; reject paths outside it."""

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    root = HOME_ROOT
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Graphiti asset path must stay inside the project folder") from exc
    return candidate


def project_relative(path: str | Path) -> str:
    """Return a stable forward-slash path relative to the package root."""

    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(HOME_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")
