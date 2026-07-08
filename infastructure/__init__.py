"""Expose `src/infastructure` as a top-level package for local module execution."""

from __future__ import annotations

from pathlib import Path


_SRC_INFRA = Path(__file__).resolve().parent.parent / "src" / "infastructure"
__path__ = [str(_SRC_INFRA)]
