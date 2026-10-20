from __future__ import annotations

import argparse
import json
import sys
import traceback

EXIT_CLEAN, EXIT_FOUND, EXIT_ERROR = 0, 1, 2


class FileFailed(RuntimeError):
    pass


def run_check(paths: list[str], since: str | None = None) -> int:
    from files import PROJECT, GitError, common_ancestor, list_covered, list_tree, parallel, read_text, resolve
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
        if since is not None:
            from hooks import added_since

            rev = resolve(PROJECT, since)
            if rev is None:
                raise GitError(f"{since!r} is not a commit in this repository")
            base = common_ancestor(PROJECT, rev)
            if base is None:
                raise GitError(f"{since!r} and HEAD have no common ancestor")
            found = added_since(PROJECT, base)
        else:
            files = list_covered(paths) if paths else list_tree()
            found = [entry for entries in parallel(judge, files) for entry in entries]
    except (ToolMissing, FileNotFoundError, FileFailed, GitError) as e:
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
    check.add_argument(
        "--since",
        metavar="REV",
        help="report only the comments the working tree added since its merge base with REV (what CI checks)",
    )
    commands.add_parser("hook-edit", help="Claude Code PreToolUse hook: deny an Edit or Write that adds a comment")
    commands.add_parser("hook-stop", help="Claude Code Stop hook: block once if the branch added a comment")
    args = parser.parse_args(argv)
    if args.command == "hook-edit":
        return run_hook("hook_edit")
    if args.command == "hook-stop":
        return run_hook("hook_stop")
    if args.since is not None and args.paths:
        parser.error("--since takes no PATH: it judges every thorn file the tree changed since its merge base with REV")
    try:
        return run_check(args.paths, args.since)
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
