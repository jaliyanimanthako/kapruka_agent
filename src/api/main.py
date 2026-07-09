"""HTTP API for the Kapruka assistant orchestration layer."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.orchestrator import KaprukaOrchestrator, build_orchestrator


class ChatRequest(BaseModel):
    """Frontend-facing chat payload."""

    message: str = Field(..., min_length=1)
    user_id: str = Field(default="demo-user", min_length=1)
    session_id: str = Field(default="demo-session", min_length=1)
    recipient_id: Optional[str] = None
    recipient_name: str = ""
    relationship: str = ""
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    use_database_short_term: bool = False


class ChatResponse(BaseModel):
    """Structured response suitable for frontend debug panels."""

    answer: str
    route: str
    reasoning: str
    progress: List[str]
    route_decision: Dict[str, Any]
    specialist_output: Dict[str, Any]
    timings_ms: Dict[str, int]


class HealthResponse(BaseModel):
    status: str


def _cors_origins() -> List[str]:
    configured = os.getenv("API_CORS_ORIGINS", "").strip()
    if configured:
        return [origin.strip() for origin in configured.split(",") if origin.strip()]
    return ["*"]


def _cors_origin_regex() -> Optional[str]:
    configured = os.getenv("API_CORS_ORIGIN_REGEX", "").strip()
    return configured or None


app = FastAPI(
    title="Kapruka Assistant API",
    version="0.1.0",
    description="FastAPI wrapper around the Kapruka memory and specialist agents.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=_cors_origin_regex(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_orchestrators: Dict[bool, KaprukaOrchestrator] = {}


def _get_orchestrator(use_database_short_term: bool) -> KaprukaOrchestrator:
    """Reuse orchestrators so in-process short-term session state is preserved."""
    if use_database_short_term not in _orchestrators:
        _orchestrators[use_database_short_term] = build_orchestrator(
            use_database_short_term=use_database_short_term,
        )
    return _orchestrators[use_database_short_term]


@app.get("/health", response_model=HealthResponse)
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> Dict[str, Any]:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    progress: List[str] = []
    orchestrator = _get_orchestrator(request.use_database_short_term)

    try:
        result = orchestrator.handle_message(
            user_message=message,
            user_id=request.user_id,
            session_id=request.session_id,
            recipient_id=request.recipient_id,
            recipient_name=request.recipient_name,
            relationship=request.relationship,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            progress_callback=progress.append,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"chat handling failed: {exc}") from exc

    return {
        "answer": result.answer,
        "route": result.route,
        "reasoning": result.reasoning,
        "progress": progress,
        "route_decision": result.route_decision,
        "specialist_output": result.specialist_output,
        "timings_ms": result.timings_ms,
    }


def create_app() -> FastAPI:
    """Factory for ASGI servers and tests."""
    return app
