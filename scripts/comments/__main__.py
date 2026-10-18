import argparse
import os
import sys
import traceback

from files import default_root, list_covered, read_text, write_text


def run_check(paths: list[str], full: bool) -> int:
    from report import report
    from rules import check

    found = []
    for f in list_covered(paths or [default_root()]):
        found.extend((f.path, v) for v in check(read_text(f.path), f.lang))
    print(report(found, full=full))
    return 1 if found else 0


def run_strip(paths: list[str]) -> int:
    from strip import Refused, strip

    changed = refused = 0
    for f in list_covered(paths):
        text = read_text(f.path)
        try:
            stripped = strip(text, f.lang)
        except Refused as e:
            print(f"{f.path}:{e.line}: strip refused: {e.reason}", file=sys.stderr)
            refused += 1
            continue
        if stripped != text:
            write_text(f.path, stripped)
            changed += 1
    print(f"strip: {changed} files changed, {refused} refused")
    return 1 if refused else 0


def run_verify(before: str, after: str, flesh: str | None) -> int:
    from verify import ToolMissing, verify_revs

    try:
        return verify_revs(before, after, flesh)
    except ToolMissing as e:
        print(f"verify: {e}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 scripts/comments",
        description="Find, strip and verify the removal of comments in thorn sources, .ccl and .par files.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="report comments outside the allowed exceptions")
    check.add_argument("--full", action="store_true", help="print each violating comment in full")
    check.add_argument("paths", nargs="*", metavar="PATH", help="files or directories (default: the repository)")
    strip = commands.add_parser("strip", help="delete exactly the comments check reports")
    strip.add_argument("paths", nargs="+", metavar="PATH")
    verify = commands.add_parser("verify", help="prove that BEFORE..AFTER changed only comments")
    verify.add_argument("before", metavar="BEFORE")
    verify.add_argument("after", metavar="AFTER")
    verify.add_argument("--flesh", default=os.environ.get("CACTUSX"), help="Cactus flesh root (default: $CACTUSX)")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            return run_check(args.paths, args.full)
        if args.command == "strip":
            return run_strip(args.paths)
        return run_verify(args.before, args.after, args.flesh)
    except FileNotFoundError as e:
        print(f"{parser.prog} {args.command}: {e}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
