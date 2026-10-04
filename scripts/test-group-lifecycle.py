#!/usr/bin/env python3
"""Check persistent history, lifecycle controls, and expected failure diagnostics."""

import argparse
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import tempfile


READ_COMPLETED = "TestGroupLifecycle: current-timelevel read completed"
CASES = (
    ("updated-read", True, (re.escape(READ_COMPLETED),)),
    (
        "checkpoint-disabled",
        False,
        (
            re.escape(
                'Group "TestGroupLifecycle::uncheckpointed_state" is integrated '
                'by ODESolvers (rhs= tag) but has checkpoint="no".'
            ),
            re.escape("Integrated groups must be checkpointed"),
        ),
    ),
    (
        "premature-read",
        False,
        (
            r'TestGroupLifecycle_ReadCurrent checking input: Grid function '
            r'"TestGroupLifecycle::evolved" is invalid on patch 0, refinement '
            r'level 0, time level 0;',
            re.escape("CycletimeLevels (invalidate current time level)"),
        ),
    ),
    (
        "history-moving",
        True,
        (re.escape("Persistent history checks passed (mask 7)"),),
    ),
    (
        "history-checkpoint",
        True,
        (re.escape("Persistent history checks passed (mask 7)"),),
    ),
    (
        "history-recover",
        True,
        (re.escape("Persistent history checks passed (mask 15)"),),
    ),
)


def run_case(command, directory, timeout):
    """Keep MPI ranks in one process group so a timeout stops the whole run."""
    logfile = directory / "run.log"
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("CACTUS_NUM_THREADS", environment["OMP_NUM_THREADS"])
    with logfile.open("w") as output:
        output.write("Command: " + shlex.join(command) + "\n")
        output.flush()
        with subprocess.Popen(
            command,
            cwd=directory,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=environment,
        ) as process:
            try:
                return process.wait(timeout=timeout), False
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return process.returncode, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "executable", type=Path,
        help="Cactus executable with TestGroupLifecycle and TestPersistentGroups",
    )
    parser.add_argument("--nprocs", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--launcher", default="mpiexec", help="MPI launcher and optional flags (default: mpiexec)"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Log directory; defaults to a retained temporary directory"
    )
    parser.add_argument("--timeout", type=float, default=120, help="Seconds per fixture")
    args = parser.parse_args()

    executable = args.executable.resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        parser.error(f"not an executable file: {executable}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    launcher = shlex.split(args.launcher)
    if not launcher:
        parser.error("--launcher must not be empty")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else Path(tempfile.mkdtemp(prefix=f"carpetx-lifecycle-{args.nprocs}proc-"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    fixtures = repository / "TestGroupLifecycle" / "fixtures"
    history_fixtures = repository / "TestPersistentGroups" / "par"
    # A fresh checkpoint directory prevents a reused --output-dir from making
    # recovery pass with files produced by an earlier invocation.
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="checkpoints-", dir=output_dir))
    # Always use the MPI launcher, including one-rank runs, to exercise the
    # same executable initialization and abort path as the normal testsuite.
    prefix = [*launcher, "-n", str(args.nprocs), str(executable)]
    passed = True
    print(f"Lifecycle test logs: {output_dir}", flush=True)
    for name, expect_success, patterns in CASES:
        directory = output_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        if name.startswith("history-"):
            template = (history_fixtures / f"{name}.par").read_text()
            rendered = template.replace("@OUTPUT_DIR@", str(directory / "output"))
            rendered = rendered.replace("@CHECKPOINT_DIR@", str(checkpoint_dir))
            parameter_file = directory / f"{name}.par"
            parameter_file.write_text(rendered)
        else:
            parameter_file = fixtures / f"{name}.par"
        command = [*prefix, str(parameter_file)]
        try:
            returncode, timed_out = run_case(command, directory, args.timeout)
        except OSError as error:
            print(f"FAIL {name}: could not launch executable: {error}", flush=True)
            return 1

        log = (directory / "run.log").read_text(errors="replace")
        normalized = " ".join(log.split())
        missing = [pattern for pattern in patterns if not re.search(pattern, normalized, re.I)]
        accepted_read = not expect_success and READ_COMPLETED.lower() in normalized.lower()
        ok = (
            not timed_out
            and (returncode == 0) == expect_success
            and not missing
            and not accepted_read
        )
        if ok:
            print(f"PASS {name} ({args.nprocs} MPI rank(s))", flush=True)
            continue

        passed = False
        reasons = []
        if timed_out:
            reasons.append("timed out")
        if (returncode == 0) != expect_success:
            reasons.append(f"unexpected exit status {returncode}")
        if missing:
            reasons.append("missing required diagnostic: " + "; ".join(missing))
        if accepted_read:
            reasons.append("invalid current-timelevel read reached the routine body")
        print(f"FAIL {name}: {', '.join(reasons)}", flush=True)
        print(f"  log: {directory / 'run.log'}", flush=True)
        print("\n".join(log.splitlines()[-30:]), flush=True)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
