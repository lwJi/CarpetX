from __future__ import annotations

import argparse
import sys
import traceback

EXIT_CLEAN, EXIT_FOUND, EXIT_ERROR = 0, 1, 2


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 scripts/boundary",
        description="A thorn reaches CarpetX only through its public headers (see agent_docs/carpetx-public-api.md).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser(
        "check",
        help="list every #include into CarpetX/src, every CARPETX_INTERNAL outside its guards, "
        "every validity kernel or why_valid_t setter named outside CarpetX/src/valid.cxx and valid.hxx, "
        "every directions parameter in the sync layer outside the flesh-facing SyncGroupsOnActiveLevels, "
        "every addition to firstvarindex or CCTK_FirstVarIndexI under CarpetX/src outside GroupRecord::var, "
        "every active_levels_t under CarpetX/src, "
        "every active_levels, tail_bin or recovery_traversal in a private CarpetX file "
        "outside traversal.cxx and traversal.hxx, "
        "every bare int group, variable, time level, patch or "
        "level parameter in a CarpetX header outside the flesh- and AMReX-fixed signatures and the window bounds, "
        "and every ghext->, driver(), GHExt, LevelData, GroupData or driver.hxx include in a checkpoint codec file",
    )
    check.add_argument("paths", nargs="*", metavar="PATH", help="files or directories to check (default: the whole tree)")
    check.add_argument(
        "--since",
        metavar="REV",
        help="report only the lines the working tree added since its merge base with REV (what CI checks)",
    )
    args = parser.parse_args(argv)
    if args.since is not None and args.paths:
        parser.error("--since takes no PATH: it judges every thorn file the tree changed since its merge base with REV")
    try:
        return run_check(args.paths, args.since)
    except Exception:
        traceback.print_exc()
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
