"""
Command-line interface — the same engine without Streamlit.

    python cli.py                                   interactive session
    python cli.py --input "SOC 0.3%, rainfall low"  single query
    python cli.py --json --input '{"soc": 0.3}'     structured in, JSON out
"""

from __future__ import annotations

import argparse

from core.conversation import Session
from core.render import to_json_string, to_markdown
from core.retrieve import kb_is_ready


def main() -> None:
    parser = argparse.ArgumentParser(description="Darukaa.Earth biodiversity intelligence")
    parser.add_argument("--input", type=str, help="Single query (text or JSON). Omit for interactive mode.")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON instead of markdown.")
    args = parser.parse_args()

    if not kb_is_ready():
        raise SystemExit("Knowledge base not built. Run: python -m knowledge.ingest")

    session = Session()

    if args.input:
        response = session.process(args.input)
        print(to_json_string(response) if args.json else to_markdown(response))
        return

    print("Darukaa.Earth biodiversity intelligence. Describe your land. 'reset' to clear, 'quit' to exit.\n")
    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in {"quit", "exit"}:
            break
        if not line:
            continue
        response = session.process(line)
        print()
        print(to_json_string(response) if args.json else to_markdown(response))
        print()


if __name__ == "__main__":
    main()
