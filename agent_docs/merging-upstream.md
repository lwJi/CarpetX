# How to merge upstream EinsteinToolkit/CarpetX main

Thorn sources, `.ccl` and `.par` files on `dev` carry no comments, so an upstream merge strips `ET/main` on a side branch first and then merges that branch. `ET/main` has no `scripts/comments`, so the tool always runs from this checkout against a worktree of upstream. Run everything below from the repo root of a `dev` checkout.

The merge base is still the last upstream commit, comments included. Wherever fork code sits next to an upstream comment, git reports a conflict even though upstream only deleted the comment. `python3 scripts/comments remerge` redoes the three-way merge of each such file with the base, ours and theirs all stripped, which leaves only the real code-vs-code conflicts.

```bash
git fetch ET
git worktree add --no-track -b strip-upstream ../carpetx-strip ET/main
python3 scripts/comments strip ../carpetx-strip
git -C ../carpetx-strip commit -am "Strip comments from EinsteinToolkit/CarpetX main $(git rev-parse --short ET/main)"
python3 scripts/comments verify ET/main strip-upstream
git worktree remove ../carpetx-strip
git switch -c merge-upstream dev
git merge --no-ff -m "Merge EinsteinToolkit/CarpetX main into dev" strip-upstream
python3 scripts/comments remerge
git commit --no-edit
python3 scripts/comments check
git rev-parse strip-upstream >> .git-blame-ignore-revs
git add .git-blame-ignore-revs
git commit -m "Ignore the EinsteinToolkit/CarpetX main strip commit in blame"
./agent_scripts/build.sh && ./agent_scripts/test.sh
git push -u origin merge-upstream
gh pr create --base dev
```

What each step expects, and what to do when it says something else:

- **How much is new.** After the fetch, `git rev-list --count dev..ET/main` is the number of upstream commits the merge brings. At 0 there is nothing to merge.
- **`--no-track`** keeps `strip-upstream` from tracking `ET/main`. The strip commit never reaches `ET/main`, and `git branch -d` checks a tracking branch against its upstream rather than HEAD, so the cleanup at the end would refuse to delete it.
- **`strip` exits 1** and prints `<file>:<line>: strip refused: …` for a constant `#if 0`/`#if 1` guard it can't resolve mechanically, such as one with an `#elif` (upstream's `Arith/src/simd.hxx` has one, in `mask_storeu`). It leaves that file untouched. Resolve the file's constant guards with `unifdef -k -o <file> <file>` (it exits 1 because it changed the file), rerun `python3 scripts/comments strip <file>`, and only then commit. `verify` resolves the old version with the same `unifdef -k`, so it covers the edit. For `simd.hxx` this gives the same file as the `Arith: remove comments` commit.
- **`verify`** must end in `only comments` and exit 0. A `FAIL` line means the strip changed more than comments: don't merge, fix the tool. Its output goes in the PR body.
- **`git merge`** exits 1 and prints one `CONFLICT` line per conflicted file, `CONFLICT (content)` for the kind `remerge` can clear. Record the count of `git diff --name-only --diff-filter=U` for the PR body.
- **`remerge`** runs straight after `git merge`, before any conflicted file is edited, because it rewrites them. It touches only unmerged covered files (C/C++ and Fortran under `src/`, `.ccl`, `.par`) that have a base, ours and theirs version and still hold conflict markers. Each one either merges clean and is staged (`merged`), or keeps only its real conflicts, with markers (`conflict`). Every other unmerged path is listed as `skipped` with the reason, and left as git left it. It exits 0 when no unmerged path is left. `git checkout -m -- <file>` brings back git's own conflict for a file it left conflicted.
- **What is left** is code against code, or a path `remerge` skipped. Resolve it as in any merge, `git add` it, and `git commit --no-edit` keeps the merge subject.
- **`check`** must exit 0. A violation here came in through a hand resolution.
- **`.git-blame-ignore-revs`** gets the strip commit, because the strip rewrites every line that lost a trailing comment. `git blame --ignore-revs-file .git-blame-ignore-revs` then skips it, and GitHub's blame view reads the file on its own.
- **The PR** body holds the `verify` output, the conflict count from `git merge` and how many `remerge` cleared, and the testsuite result. Merge it with a merge commit (`gh pr merge <N> --merge`): a squash or rebase rewrites the strip commit, and its SHA in `.git-blame-ignore-revs` would then name nothing.
- **Cleanup**, once the PR is merged: `git switch dev && git pull && git branch -d merge-upstream strip-upstream`.
