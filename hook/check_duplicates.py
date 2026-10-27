"""Claude Code PostToolUse hook (Edit|Write|MultiEdit): flag Java methods that semantically duplicate existing code.

Exit 2 + stderr feeds the message back to Claude so it can reuse/refactor instead of duplicating.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.hook_engine import find_duplicates


def main():
    data = json.load(sys.stdin)
    path = (data.get("tool_input") or {}).get("file_path", "")
    project = data.get("cwd") or "."
    if not path.endswith(".java") or not Path(path).exists():
        return 0
    hits = find_duplicates(project, path)
    if not hits:
        return 0
    lines = ["Semantic duplication detected in the code you just wrote. Reuse or extract shared logic instead of duplicating:"]
    for h in hits:
        lines.append(f"- {h['method']} (lines {h['lines']}) is ~{h['sim']:.0%} similar to {h['similar_to']}")
    print("\n".join(lines), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
