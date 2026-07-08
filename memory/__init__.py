"""Expose `src/memory` as a top-level package for local module execution."""

from __future__ import annotations

from pathlib import Path


_SRC_MEMORY = Path(__file__).resolve().parent.parent / "src" / "memory"
__path__ = [str(_SRC_MEMORY)]
