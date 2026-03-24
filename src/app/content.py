from __future__ import annotations
from pathlib import Path

MARKDOWN_DIR = Path(__file__).parent / "markdown"


def read_markdown(filename: str) -> str:
    """Read markdown file from app/markdown directory."""

    return (MARKDOWN_DIR / filename).read_text(encoding="utf-8")
