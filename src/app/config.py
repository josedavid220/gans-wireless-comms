from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return start


REPO_ROOT = find_repo_root(Path(__file__))
LOGS_ROOT = REPO_ROOT / "logs" / "mftr" / "cgan"

# Curated subset for v1 demo. Extend this list as needed.
CURATED_VERSIONS = [
    "version_16",
]

# Fixed slider bounds used in the UI.
UI_PARAM_BOUNDS = {
    "m": (1.0, 10.0),
    "mu": (1.0, 15.0),
    "K": (0.0, 20.0),
    "delta": (0.0, 0.999),
    "omega": (1.0, 2.0),
}

# Runtime defaults
DEFAULT_NUM_SAMPLES = 10_000
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 2048
