from __future__ import annotations

import argparse
import json
import sys
import traceback

EXIT_CLEAN, EXIT_FOUND, EXIT_ERROR = 0, 1, 2


class FileFailed(RuntimeError):
    pass


def run_check(paths: list[str]) -> int:
    from files import list_covered, list_tree, parallel, read_text
    from lex_clang import LexError, ToolMissing
    from report import render
    from rules import check_text

    def judge(item):
        name, path, lang = item
        try:
            return [(name, v) for v in check_text(read_text(path), lang)]
        except (LexError, UnicodeError, OSError) as e:
            raise FileFailed(f"{name}: {e}") from e

    try:
        files = list_covered(paths) if paths else list_tree()
        found = [entry for entries in parallel(judge, files) for entry in entries]
    except (ToolMissing, FileNotFoundError, FileFailed) as e:
        print(f"comments check: {e}", file=sys.stderr)
        return EXIT_ERROR
    sys.stdout.reconfigure(errors="backslashreplace")
    print(render(found))
    return EXIT_FOUND if found else EXIT_CLEAN


def run_hook(name: str) -> int:
    try:
        import hooks

        raw = sys.stdin.buffer.read()
    except Exception as e:
        print(json.dumps({"systemMessage": f"comment check skipped: {type(e).__name__}: {e}"}))
        return 0
    return getattr(hooks, name)(raw, sys.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 scripts/comments",
        description="Keep comments out of thorn files (see \"Code comments\" in CLAUDE.md).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="report every comment outside the allow-list")
    check.add_argument("paths", nargs="*", metavar="PATH", help="files or directories to check (default: the whole tree)")
    commands.add_parser("hook-edit", help="Claude Code PreToolUse hook: deny an Edit or Write that adds a comment")
    commands.add_parser("hook-stop", help="Claude Code Stop hook: block once if the branch added a comment")
    args = parser.parse_args(argv)
    if args.command == "hook-edit":
        return run_hook("hook_edit")
    if args.command == "hook-stop":
        return run_hook("hook_stop")
    try:
        return run_check(args.paths)
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
