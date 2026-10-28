from __future__ import annotations

import argparse
import json
import sys
import traceback

EXIT_CLEAN, EXIT_FOUND, EXIT_ERROR = 0, 1, 2
HOOKS = {"hook-edit": "hook_edit", "hook-stop": "hook_stop"}


def run_check(paths: list[str], since: str | None = None) -> int:
    from rules import findings, render, since_findings
    from files import PROJECT, GitError, common_ancestor, list_covered, list_tree, read_text, resolve

    try:
        if since is not None:
            rev = resolve(PROJECT, since)
            if rev is None:
                raise GitError(f"{since!r} is not a commit in this repository")
            base = common_ancestor(PROJECT, rev)
            if base is None:
                raise GitError(f"{since!r} and HEAD have no common ancestor")
            found = since_findings(PROJECT, base)
        else:
            files = list_covered(paths) if paths else list_tree()
            found = [(name, f) for name, path, lang in files for f in findings(name, read_text(path), lang)]
    except (FileNotFoundError, GitError, OSError, UnicodeError) as e:
        print(f"boundary check: {e}", file=sys.stderr)
        return EXIT_ERROR
    sys.stdout.reconfigure(errors="backslashreplace")
    print(render(found))
    return EXIT_FOUND if found else EXIT_CLEAN


def run_hook(name: str) -> int:
    try:
        import hooks

        raw = sys.stdin.buffer.read()
    except Exception as e:
        print(json.dumps({"systemMessage": f"boundary check skipped: {type(e).__name__}: {e}"}))
        return 0
    return getattr(hooks, name)(raw, sys.stdout)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 1 and argv[0] in HOOKS:
        return run_hook(HOOKS[argv[0]])
    try:
        from rules import PREAMBLE, RULE, RULES, series
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR
    parser = argparse.ArgumentParser(prog="python3 scripts/boundary", description=PREAMBLE)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check",
        help="list every " + series([rule.phrase(rule.noun) for rule in RULES]),
        description=RULE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check.add_argument("paths", nargs="*", metavar="PATH", help="files or directories to check (default: the whole tree)")
    check.add_argument(
        "--since",
        metavar="REV",
        help="report only the lines the working tree added since its merge base with REV (what CI checks)",
    )
    commands.add_parser("hook-edit", help="Claude Code PreToolUse hook: deny an Edit or Write that adds a boundary finding")
    commands.add_parser("hook-stop", help="Claude Code Stop hook: block once if the branch added a boundary finding")
    args = parser.parse_args(argv)
    if args.command in HOOKS:
        return run_hook(HOOKS[args.command])
    if args.since is not None and args.paths:
        parser.error("--since takes no PATH: it judges every thorn file the tree changed since its merge base with REV")
    try:
        return run_check(args.paths, args.since)
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
