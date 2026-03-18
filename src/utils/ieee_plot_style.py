from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Best-effort repo root finder (looks for pyproject.toml)."""
    path = (start or Path.cwd()).resolve()
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return path


def ieee_mplstyle_path(repo_root: Optional[Path] = None) -> Path:
    root = (repo_root or find_repo_root()).resolve()
    return root / "report" / "ieee.mplstyle"


@dataclass(frozen=True)
class IEEEStyleResult:
    applied: bool
    style_path: Optional[Path]


def apply_ieee_style(repo_root: Optional[Path] = None) -> IEEEStyleResult:
    """Apply the repo's IEEE-friendly Matplotlib style.

    Returns whether the style was applied and which file was used.
    """
    

    style_path = ieee_mplstyle_path(repo_root)
    if style_path.exists():
        plt.style.use(str(style_path))
        return IEEEStyleResult(applied=True, style_path=style_path)

    # Fallback (should rarely happen): keep it minimal and safe.
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "lines.linewidth": 1.6,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    return IEEEStyleResult(applied=False, style_path=None)
