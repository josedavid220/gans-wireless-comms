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

def _version_sort_key(name: str) -> tuple[int, str]:
    try:
        return (int(name.split("_", 1)[1]), name)
    except (IndexError, ValueError):
        return (-   1, name)


def discover_versions(logs_root: Path) -> list[str]:
    if not logs_root.exists():
        return []

    versions = [
        p.name
        for p in logs_root.iterdir()
        if p.is_dir() and p.name.startswith("version_")
    ]
    return sorted(versions, key=_version_sort_key)


# Auto-discovered from LOGS_ROOT.
CURATED_VERSIONS = discover_versions(LOGS_ROOT)

# Fixed slider bounds used in the UI.
UI_PARAM_BOUNDS = {
    "m": (1.0, 10.0),
    "mu": (1.0, 15.0),
    "K": (0.0, 20.0),
    "delta": (0.0, 0.999),
    "omega": (1.0, 2.0),
}

DEFAULT_TEST_PARAMS = {
    "m": 8.0,
    "mu": 5.0,
    "K": 9.0,
    "delta": 0.9,
    "omega": 1.0,
}

# Runtime defaults
DEFAULT_NUM_SAMPLES = 10_000
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 2048
