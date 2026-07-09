"""CLI demo for the Kapruka specialist orchestration layer."""

from __future__ import annotations

import argparse
import json
import sys

from agents.orchestrator import build_orchestrator
from memory.memory_ops import CognitiveMemoryStack


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one query through the Kapruka specialist orchestrator.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--user-id", default="demo-user")
    parser.add_argument("--session-id", default="demo-session")
    parser.add_argument("--recipient-id", default="")
    parser.add_argument("--recipient-name", default="")
    parser.add_argument("--relationship", default="")
    parser.add_argument("--catalog", default="catalog.json")
    parser.add_argument("--sync-catalog", action="store_true")
    parser.add_argument("--use-supabase-st", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()

    def emit_progress(message: str) -> None:
        if not args.quiet_progress:
            print(f"[progress] {message}", file=sys.stderr, flush=True)

    if args.sync_catalog:
        emit_progress("Syncing catalog into Qdrant...")
        CognitiveMemoryStack().sync_catalog(args.catalog)

    emit_progress(
        "Building orchestrator..."
        + (" using Supabase short-term memory" if args.use_supabase_st else " using local short-term memory")
    )
    orchestrator = build_orchestrator(use_database_short_term=args.use_supabase_st)

    response = orchestrator.handle_message(
        user_message=args.query,
        user_id=args.user_id,
        session_id=args.session_id,
        recipient_id=args.recipient_id,
        recipient_name=args.recipient_name,
        relationship=args.relationship,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        progress_callback=emit_progress,
    )

    print("=== ROUTE DECISION ===")
    print(json.dumps(response.route_decision, indent=2, ensure_ascii=False))
    print("\n=== TIMINGS (ms) ===")
    print(json.dumps(response.timings_ms, indent=2, ensure_ascii=False))
    catalog_output = response.specialist_output.get("catalog", {})
    if "memory_gate" in catalog_output:
        print("\n=== MEMORY GATE ===")
        print(json.dumps(catalog_output["memory_gate"], indent=2, ensure_ascii=False))
    print("\n=== SPECIALIST OUTPUT ===")
    print(json.dumps(response.specialist_output, indent=2, ensure_ascii=False))
    print("\n=== FINAL ANSWER ===")
    print(response.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
