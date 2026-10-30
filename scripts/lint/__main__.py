from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

EXIT_CLEAN, EXIT_FOUND, EXIT_ERROR = 0, 1, 2
ENFORCEMENT = """\
How the checks are run:
  The edit hook in .claude/settings.json runs hook-edit on every Edit and Write and denies one that
  adds a finding. The Stop and SubagentStop hooks run hook-stop and block the turn once if the branch
  has added a finding since dev. The Lint workflow runs `check --since` from the PR's base on every PR
  and on every push to dev. What a merge brings in from another branch is not judged, only the
  merge's own resolution. `check --since origin/dev` shows what all of them would report; the rules
  are in agent_docs/code-comments.md, agent_docs/carpetx-public-api.md and agent_docs/carpetx-internals.md.
"""
HOOKS = {"hook-edit": "hook_edit", "hook-stop": "hook_stop"}


def run_check(checks, paths: list[str], since: str | None = None) -> int:
    from files import PROJECT, GitError, common_ancestor, resolve
    from run import FileFailed, check_since, check_tree, has_findings, render, render_change

    try:
        if since is not None:
            rev = resolve(PROJECT, since)
            if rev is None:
                raise GitError(f"{since!r} is not a commit in this repository")
            base = common_ancestor(PROJECT, rev)
            if base is None:
                raise GitError(f"{since!r} and HEAD have no common ancestor")
            found = check_since(PROJECT, base, checks)
            report = render_change(found, every=True)
        else:
            found = check_tree(paths, checks)
            report = render(found)
    except (FileNotFoundError, FileFailed, GitError, OSError, UnicodeError) as e:
        print(f"lint check: {e}", file=sys.stderr)
        return EXIT_ERROR
    sys.stdout.reconfigure(errors="backslashreplace")
    print(report)
    return EXIT_FOUND if has_findings(found) else EXIT_CLEAN


def run_doc(write: bool) -> int:
    from boundary.doc import MAP, render, rewrite
    from files import PROJECT

    if not write:
        print(render(), end="")
        return EXIT_CLEAN
    path = os.path.join(PROJECT, MAP)
    try:
        with open(path) as f:
            text = f.read()
        new = rewrite(text)
    except (OSError, ValueError) as e:
        print(f"lint doc: {e}", file=sys.stderr)
        return EXIT_ERROR
    if new != text:
        with open(path, "w") as f:
            f.write(new)
    print(f"{'rewrote' if new != text else 'unchanged'} {MAP}")
    return EXIT_CLEAN


def run_hook(name: str) -> int:
    try:
        import hooks

        raw = sys.stdin.buffer.read()
    except Exception as e:
        print(json.dumps({"systemMessage": f"lint check skipped: {type(e).__name__}: {e}"}))
        return 0
    return getattr(hooks, name)(raw, sys.stdout)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 1 and argv[0] in HOOKS:
        return run_hook(HOOKS[argv[0]])
    try:
        from checks import CHECKS

        checks = CHECKS()
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR
    names = [check.name for check in checks]
    parser = argparse.ArgumentParser(
        prog="python3 scripts/lint",
        description="The checks the hooks and CI hold a change to: " + ", ".join(names) + ".",
        epilog=ENFORCEMENT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check",
        help="list every comment line outside the allow-list and every line a boundary rule reports",
        description="\n\n".join(check.rule for check in checks),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check.add_argument("paths", nargs="*", metavar="PATH", help="files or directories to check (default: the whole tree)")
    check.add_argument(
        "--since",
        metavar="REV",
        help="report only what the working tree added or touched since its merge base with REV (what CI checks)",
    )
    check.add_argument(
        "--only",
        action="append",
        choices=names,
        metavar="CHECK",
        help="run only this check, one of " + ", ".join(names) + " (repeatable; default: every check)",
    )
    doc = commands.add_parser("doc", help="print the generated blocks of agent_docs/carpetx-internals.md: the stack and the includes")
    doc.add_argument("--write", action="store_true", help="rewrite them in agent_docs/carpetx-internals.md instead")
    commands.add_parser("hook-edit", help="Claude Code PreToolUse hook: deny an Edit or Write that adds a finding")
    commands.add_parser("hook-stop", help="Claude Code Stop hook: block once if the branch added a finding")
    args = parser.parse_args(argv)
    if args.command in HOOKS:
        return run_hook(HOOKS[args.command])
    if args.command == "doc":
        return run_doc(args.write)
    if args.since is not None and args.paths:
        parser.error("--since takes no PATH: it judges every thorn file the tree changed since its merge base with REV")
    if args.only:
        checks = tuple(check for check in checks if check.name in args.only)
    try:
        return run_check(checks, args.paths, args.since)
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
