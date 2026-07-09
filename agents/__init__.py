"""Compatibility shim so `python -m agents...` works from the repo root."""

from pathlib import Path

_SRC_AGENTS = Path(__file__).resolve().parent.parent / "src" / "agents"
__path__ = [str(_SRC_AGENTS)]
