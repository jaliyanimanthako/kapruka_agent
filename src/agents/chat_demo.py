"""Interactive chat CLI for testing the Kapruka memory stack."""

from __future__ import annotations

import argparse
import json
import sys

from agents.orchestrator import build_orchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an interactive Kapruka chat with memory.")
    parser.add_argument("--user-id", default="demo-user")
    parser.add_argument("--session-id", default="demo-session")
    parser.add_argument("--recipient-id", default="")
    parser.add_argument("--recipient-name", default="")
    parser.add_argument("--relationship", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--use-supabase-st", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    parser.add_argument("--show-debug", action="store_true")
    args = parser.parse_args()

    def emit_progress(message: str) -> None:
        if not args.quiet_progress:
            print(f"[progress] {message}", file=sys.stderr, flush=True)

    emit_progress(
        "Building interactive orchestrator..."
        + (" using Supabase short-term memory" if args.use_supabase_st else " using local short-term memory")
    )
    orchestrator = build_orchestrator(use_database_short_term=args.use_supabase_st)

    print("Interactive Kapruka chat started.")
    print("Commands: /debug on, /debug off, /exit")

    show_debug = args.show_debug

    while True:
        try:
            user_message = input("\nYou: ").strip()
        except EOFError:
            print()
            break

        if not user_message:
            continue
        if user_message.lower() in {"/exit", "exit", "quit"}:
            break
        if user_message.lower() == "/debug on":
            show_debug = True
            print("Debug output enabled.")
            continue
        if user_message.lower() == "/debug off":
            show_debug = False
            print("Debug output disabled.")
            continue

        response = orchestrator.handle_message(
            user_message=user_message,
            user_id=args.user_id,
            session_id=args.session_id,
            recipient_id=args.recipient_id,
            recipient_name=args.recipient_name,
            relationship=args.relationship,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            progress_callback=emit_progress,
        )

        print(f"Assistant: {response.answer}")

        if show_debug:
            print("\n--- DEBUG ---")
            print("Route:")
            print(json.dumps(response.route_decision, indent=2, ensure_ascii=False))
            print("Timings:")
            print(json.dumps(response.timings_ms, indent=2, ensure_ascii=False))
            print("Specialist Output:")
            print(json.dumps(response.specialist_output, indent=2, ensure_ascii=False))

    print("Chat ended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
