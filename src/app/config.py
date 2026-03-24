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

# Broad input bounds used when extrapolation is enabled.
EXTRAPOLATION_BOUNDS = {
    "m": (1.0, 16.0),
    "mu": (1.0, 15.66),
    "K": (0.0, 20.0),
    "delta": (0.0, 0.999),
    "omega": (0.2, 4.0),
}

# Runtime defaults
DEFAULT_NUM_SAMPLES = 10_000
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 512
