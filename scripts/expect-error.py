#!/usr/bin/env python3
"""Run expected-failure parameter files.

Each DIR holds pairs <name>.par / <name>.expect. Every .par is run with EXE
from a fresh temporary directory; it passes when the run exits non-zero and
its combined stdout/stderr contains the text of the .expect file. The
testsuite globs only <Thorn>/test/*.par, so a subdirectory such as
<Thorn>/test/errors is where a parfile that must abort lives.

The run's environment holds EXPECT_ERROR_PARFILE_DIR, the absolute directory
of the .par, so that a parfile can name a file beside it, for example
IO::recover_dir = $ENV{EXPECT_ERROR_PARFILE_DIR} + "/../checkpoints".

Exit 0 when every parfile fails as expected, 1 when one does not, 2 on a
usage or setup error.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import tempfile

EXIT_CLEAN, EXIT_FOUND, EXIT_ERROR = 0, 1, 2
TAIL_LINES = 25


def collect(dirs: list[str]) -> list[tuple[str, str]]:
    cases = []
    for d in dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"no such directory: {d}")
        pars = sorted(glob.glob(os.path.join(d, "*.par")))
        if not pars:
            raise FileNotFoundError(f"no .par file in {d}")
        for par in pars:
            expect = par[: -len(".par")] + ".expect"
            if not os.path.isfile(expect):
                raise FileNotFoundError(f"{par} has no {os.path.basename(expect)} beside it")
            cases.append((par, expect))
    return cases


def run_case(exe: str, par: str, expect: str, timeout: float) -> tuple[bool, str]:
    with open(expect, encoding="utf-8") as f:
        wanted = f.read().strip()
    if not wanted:
        raise ValueError(f"{expect} is empty")
    par = os.path.abspath(par)
    env = dict(os.environ, EXPECT_ERROR_PARFILE_DIR=os.path.dirname(par))
    with tempfile.TemporaryDirectory() as cwd:
        try:
            proc = subprocess.run(
                [exe, par],
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode("utf-8", "replace")
            return False, f"timed out after {timeout:g} s\n{out}"
    out = proc.stdout.decode("utf-8", "replace")
    if proc.returncode == 0:
        return False, f"exited 0, but the run must abort with: {wanted}\n{out}"
    if wanted not in out:
        return False, f"exited {proc.returncode} without the expected text: {wanted}\n{out}"
    return True, f"exited {proc.returncode} with: {wanted}"


def tail(text: str) -> str:
    lines = text.rstrip("\n").split("\n")
    if len(lines) > TAIL_LINES:
        lines = [f"... ({len(lines) - TAIL_LINES} lines omitted)"] + lines[-TAIL_LINES:]
    return "\n".join("    " + line for line in lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 scripts/expect-error.py",
        description="Run every <name>.par in DIR and require it to abort with the text of <name>.expect.",
    )
    parser.add_argument("exe", help="the Cactus executable")
    parser.add_argument("dirs", nargs="+", metavar="DIR", help="directories of .par/.expect pairs")
    parser.add_argument("--timeout", type=float, default=600, metavar="SECONDS", help="per parfile (default: 600)")
    args = parser.parse_args(argv)

    exe = os.path.abspath(args.exe)
    if not (os.path.isfile(exe) and os.access(exe, os.X_OK)):
        print(f"expect-error: {args.exe} is not an executable", file=sys.stderr)
        return EXIT_ERROR
    try:
        cases = collect(args.dirs)
    except FileNotFoundError as e:
        print(f"expect-error: {e}", file=sys.stderr)
        return EXIT_ERROR

    failures = 0
    for par, expect in cases:
        try:
            ok, detail = run_case(exe, par, expect, args.timeout)
        except (OSError, ValueError) as e:
            print(f"expect-error: {par}: {e}", file=sys.stderr)
            return EXIT_ERROR
        mark = "✓" if ok else "✗"
        print(f"{mark} {par}: {detail.splitlines()[0]}")
        if not ok:
            failures += 1
            print(tail("\n".join(detail.splitlines()[1:])))
    total = len(cases)
    if failures:
        print(f"✗ expect-error: {failures} of {total} parfile{'s' if total != 1 else ''} did not fail as expected")
        return EXIT_FOUND
    print(f"✓ expect-error: {total} parfile{'s' if total != 1 else ''} failed as expected")
    return EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
