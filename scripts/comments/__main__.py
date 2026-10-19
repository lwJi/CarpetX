from __future__ import annotations

import argparse
import json
import sys
import traceback

EXIT_CLEAN, EXIT_FOUND, EXIT_ERROR = 0, 1, 2


def run_check(paths: list[str]) -> int:
    from files import list_covered, read_text
    from lex_clang import ToolMissing
    from report import render
    from rules import check_text

    try:
        found = []
        for name, path, lang in list_covered(paths):
            found.extend((name, v) for v in check_text(read_text(path), lang))
    except (ToolMissing, FileNotFoundError) as e:
        print(f"comments check: {e}", file=sys.stderr)
        return EXIT_ERROR
    sys.stdout.reconfigure(errors="backslashreplace")
    print(render(found))
    return EXIT_FOUND if found else EXIT_CLEAN


def run_hook_edit() -> int:
    try:
        import hooks

        raw = sys.stdin.buffer.read()
    except Exception as e:
        print(json.dumps({"systemMessage": f"comment check skipped: {type(e).__name__}: {e}"}))
        return 0
    return hooks.hook_edit(raw, sys.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 scripts/comments",
        description="Keep comments out of thorn files (see \"Code comments\" in CLAUDE.md).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="report every comment outside the allow-list")
    check.add_argument("paths", nargs="+", metavar="PATH", help="files or directories to check")
    commands.add_parser("hook-edit", help="Claude Code PreToolUse hook: deny an Edit or Write that adds a comment")
    args = parser.parse_args(argv)
    if args.command == "hook-edit":
        return run_hook_edit()
    try:
        return run_check(args.paths)
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
