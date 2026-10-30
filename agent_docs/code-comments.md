# Code comments in thorn files

Comments in thorn files can only be deleted.

## Which files are covered

A thorn is a top-level directory with an `interface.ccl`, and its files are the C, C++ and Fortran sources under `src/`, every `.ccl`, every `.par` and `.par-DISABLED`, and `make.code.defn`. Repo tooling (`scripts/`, `agent_scripts/`, `docker/`, workflows, option lists) is not covered, and neither is `BenchX/`.

## What a change may do

Don't add a comment line, and don't change one: a reworded, appended, moved or copied line is a new line. When your change edits the code under a comment, or deletes some of the comment's lines, delete the whole comment in the same change. The code under a comment runs from the comment to the next blank line or the next comment, but neither ends it inside a block that code opens (a function, class, `if` or loop, but not a namespace or an `extern "C"` block): it runs through the block's closing brace and on to the next blank line or comment after it. A comment on a line of code covers that line. Spacing doesn't count, so re-indenting is not a change. Moving code takes it out from under its comment, so the comment goes too; rename a file with `git mv` to keep its comments.

## Allowed comment lines

A new comment line is allowed only if it is one of these:

- A tool marker: a namespace closer such as `} // namespace CarpetX`, `// clang-format off` / `// clang-format on`, or a bare `//` (after code it steers clang-format's line breaks).
- A citation that consists of a single URL, arXiv ID or DOI, optionally introduced by `See`, `From`, `Taken from` or `cf.` and followed by an arXiv category and equation numbers, for example `// See arXiv:1111.2177 [gr-qc], (74-75)` or `// doi:10.1016/0021-9991(88)90177-5`. Write an author-year citation as its DOI.

Everything else is refused, however short: prose, TODO and FIXME, `/*arg=*/` labels, line references into other code bases, commented-out code and `.par` settings, `#if 0` / `#if 1` blocks, and compile-time messages (`#warning`, `#pragma message`). There is no way to exempt a line or a file. CCL description strings and runtime messages are not comments, but don't move comment prose into them.

## Comments already in a file

A comment that is already in a file is a claim to check against the code, never the reason for a change. The check sees only the comments over the code you change. When your change makes another comment false, such as one on a declaration in a header or one that names something you renamed, delete that one too; leave the rest alone. When part of a comment you delete is still true, put that part in code or in your commit message.

## Where explanations go

Explain in code (an `assert`, a name, a `CCTK_VERROR`) or in your commit message, never in a comment. When the code doesn't explain itself, read the file's `git log`; `git log -S '<words>'` lists the commits that added or deleted a comment line. History describes the code as it was at that commit: check it against the current code before relying on it.

## How it is enforced

`python3 scripts/lint check --since origin/dev` shows what the hooks and the `Lint` workflow would report, beside the `boundary` check that shares the tool, and `--only no-comments` shows this check alone; `python3 scripts/lint --help` says how they run it.
