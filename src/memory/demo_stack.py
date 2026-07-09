"""Simple end-to-end demo for the 3-tier Kapruka memory stack."""

from __future__ import annotations

import argparse
import json

from memory.memory_ops import CognitiveMemoryStack


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a cognitive memory stack demo.")
    parser.add_argument("--user-id", default="demo-user")
    parser.add_argument("--session-id", default="demo-session")
    parser.add_argument("--recipient-id", default="")
    parser.add_argument("--recipient-name", default="")
    parser.add_argument("--relationship", default="")
    parser.add_argument("--preference", action="append", default=[])
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument("--turn", action="append", default=[])
    parser.add_argument("--query", required=True)
    parser.add_argument("--catalog", default="catalog.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    stack = CognitiveMemoryStack()

    for content in args.turn:
        stack.add_turn(args.user_id, args.session_id, "user", content)

    if args.recipient_id and (args.preference or args.note or args.relationship or args.recipient_name):
        stack.save_recipient_profile(
            recipient_id=args.recipient_id,
            name=args.recipient_name,
            relationship=args.relationship,
            preferences=args.preference,
            notes=args.note,
        )

    stack.sync_catalog(args.catalog)
    bundle = stack.build_context_bundle(
        user_id=args.user_id,
        session_id=args.session_id,
        query=args.query,
        recipient_id=args.recipient_id,
        top_k=args.top_k,
    )
    print(json.dumps(bundle, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
