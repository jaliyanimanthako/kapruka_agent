"""Compatibility shim so `uvicorn api.main:app` works from the repo root."""

from pathlib import Path

_SRC_API = Path(__file__).resolve().parent.parent / "src" / "api"
__path__ = [str(_SRC_API)]
