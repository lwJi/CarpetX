from __future__ import annotations

import argparse
import json
import os
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


def run_strip(paths: list[str]) -> int:
    from files import list_covered, parallel, read_text
    from lex_clang import LexError, ToolMissing
    from strip import StripError, strip

    def stripped(item):
        name, path, lang = item
        try:
            text = read_text(path)
            return name, path, text, strip(text, lang)
        except (LexError, StripError, UnicodeError, OSError) as e:
            raise FileFailed(f"{name}: {e}") from e

    try:
        results = parallel(stripped, list_covered(paths))
    except (ToolMissing, FileNotFoundError, FileFailed) as e:
        print(f"comments strip: {e}", file=sys.stderr)
        return EXIT_ERROR
    changed = 0
    for name, path, text, new in results:
        if new == text:
            continue
        with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
            f.write(new)
        changed += 1
        print(name)
    noun = "file" if len(results) == 1 else "files"
    print(f"stripped {changed} of {len(results)} covered {noun}")
    return EXIT_CLEAN


def run_verify(command: str, revs: list[str], flesh: str | None) -> int:
    from files import PROJECT, GitError, resolve
    from lex_clang import LexError, ToolMissing
    from strip import StripError
    from upstream import verify_skips
    from verify import report, verify_revs

    sys.stdout.reconfigure(errors="backslashreplace")
    try:
        for rev in revs:
            if resolve(PROJECT, rev) is None:
                raise ValueError(f"no such commit: {rev}")
        if command == "verify":
            ok = report(verify_revs(PROJECT, revs[0], revs[1], flesh), f"verify {revs[0]} {revs[1]}", print)
        else:
            ok = verify_skips(PROJECT, revs[0], flesh, print)
    except (ToolMissing, GitError, LexError, StripError, ValueError) as e:
        print(f"comments {command}: {e}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_CLEAN if ok else EXIT_FOUND


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
    strip = commands.add_parser("strip", help="remove every comment outside the allow-list, and fold constant #if guards")
    strip.add_argument("paths", nargs="+", metavar="PATH", help="files or directories to strip in place")
    flesh = {"metavar": "DIR", "default": os.environ.get("CACTUSX"),
             "help": "the Cactus flesh, for its .ccl and .par grammars (default: $CACTUSX)"}
    verify = commands.add_parser("verify", help="prove that AFTER differs from BEFORE only in comments")
    verify.add_argument("--flesh", **flesh)
    verify.add_argument("before", metavar="BEFORE")
    verify.add_argument("after", metavar="AFTER")
    skips = commands.add_parser("verify-skips", help="verify each commit added to .git-blame-ignore-revs since BASE")
    skips.add_argument("--flesh", **flesh)
    skips.add_argument("base", metavar="BASE")
    commands.add_parser("hook-edit", help="Claude Code PreToolUse hook: deny an Edit or Write that adds a comment")
    commands.add_parser("hook-stop", help="Claude Code Stop hook: block once if the branch added a comment")
    args = parser.parse_args(argv)
    if args.command == "hook-edit":
        return run_hook("hook_edit")
    if args.command == "hook-stop":
        return run_hook("hook_stop")
    try:
        if args.command == "strip":
            return run_strip(args.paths)
        if args.command == "verify":
            return run_verify("verify", [args.before, args.after], args.flesh)
        if args.command == "verify-skips":
            return run_verify("verify-skips", [args.base], args.flesh)
        return run_check(args.paths)
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
