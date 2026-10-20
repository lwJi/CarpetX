# How to merge upstream EinsteinToolkit/CarpetX main

Upstream keeps its comments and `dev` has none (see "Code comments" in `CLAUDE.md`), so upstream arrives through a stripped lineage. Each upstream strip commit `S_k` holds the tree of an upstream commit `ET_k` with every thorn file put through `scripts/comments strip`. Its first parent is `ET_k`, its second parent is the previous strip commit `S_{k-1}`, and its message ends with a `Comments-Stripped-From: <ET_k>` trailer. `dev` already contains `S_{k-1}`, so git takes it as the merge base, and since the base has no comments either, the merge sees only code. The lineage starts at `S0`, the strip of `6c5914436` (the upstream commit PR #138 merged), which `dev` records with `git merge -s ours`.

`strip-upstream` needs `clang++`, `unifdef`, `perl` and the Cactus flesh (`$CACTUSX`, or `--flesh DIR`). All of them are on the host and in the sbx sandbox.

```bash
git remote get-url ET || git remote add ET https://github.com/EinsteinToolkit/CarpetX.git
git fetch ET
git fetch origin
git switch -c merge-upstream origin/dev
s=$(python3 scripts/comments strip-upstream ET/main)
git merge --no-ff -m "Merge EinsteinToolkit/CarpetX main into dev" "$s"
```

`strip-upstream` finds `S_{k-1}` as the newest commit in `HEAD`'s history with the trailer. It then strips a temporary worktree of `ET/main`, commits the result on top of `ET/main` and `S_{k-1}`, and checks it with `verify ET/main S_k`. Its progress goes to stderr, and only once `verify` passes does it print `S_k`'s SHA to stdout. If `ET/main` holds nothing that `S_{k-1}` doesn't already strip, it prints `S_{k-1}` itself, the merge reports `Already up to date.`, and you are done.

## Resolving conflicts

Neither side of the merge has comments, so every conflict is between code. Resolve each one as for any merge, so that it keeps both the fork's change and upstream's. Two cases need more than that:

- **A hunk in a file the fork reformatted.** The `<Thorn>: clang-format after removing comments` commits re-wrapped lines on `dev` that `S_k` leaves as `strip` wrote them. Where upstream touched such a line, the hunk conflicts even if the fork changed no code there. These are the files those commits changed: `git log --format= --name-only --grep=': clang-format after removing comments$' | sort -u`. If the two sides of the hunk differ only in line breaks and spacing (they are equal once every space and line break is removed), keep `dev`'s side. If upstream changed code in the hunk, take upstream's side first and then reapply any fork change inside it, so that upstream's change is kept. Either way, finish by running `clang-format -i <file>` on the merged file, after checking that `clang-format --version` prints 22.1.1 (otherwise `python3 -m pip install clang-format==22.1.1` and put its bin directory first on `PATH`).
- **A compile-time message.** `strip` leaves `#warning` and `#pragma message` lines in place, so one that upstream added comes through the merge. After the merge, remove it by hand in a commit of its own, and say in that commit's message what it said.

Conclude the merge with `git commit` if it stopped. Then confirm that the tree has no comments:

```bash
python3 scripts/comments check
```

It must print `0 comments`.

## Hiding the strip commit from blame

Add `S_k` to `.git-blame-ignore-revs` in a commit of its own, and prove it the way the `no-comments` check will. `verify-skips` confirms that `S_k`'s first parent is the commit its trailer names, and that `verify S_k^1 S_k` passes:

```bash
{ echo; git log -1 --format='# %s%n%H' "$s"; } >> .git-blame-ignore-revs
git commit -m "Ignore the upstream strip commit in blame" .git-blame-ignore-revs
python3 scripts/comments verify-skips origin/dev
```

## Building, testing and merging the PR

```bash
./agent_scripts/build.sh && ./agent_scripts/test.sh
git push -u origin merge-upstream
gh pr create --base dev --title "Merge EinsteinToolkit/CarpetX main into dev"
```

Merge the PR with a merge commit (`gh pr merge <N> --merge`), never by squash or rebase. Either of those would drop `S_k` from `dev`'s history, and with it the next merge's base and the commit that `.git-blame-ignore-revs` lists.

## When `strip` changes

`strip-upstream` assumes that `strip` still writes what it wrote for `S_{k-1}`. If a change to `scripts/comments` changes `strip`'s output, re-seed before the next upstream merge, from the upstream commit that the newest strip commit names, and add the new seed to `.git-blame-ignore-revs` as above:

```bash
et=$(git rev-parse "$(git log -1 --format=%H -E --grep='^Comments-Stripped-From: [0-9a-f]{40}')^1")
s=$(python3 scripts/comments strip-upstream --seed "$et")
git merge -s ours -m "Seed the stripped upstream lineage again at $(git rev-parse --short "$et")" "$s"
```

Changes headed upstream go the other way by hand: port them onto a branch cut from `ET/main`, and keep upstream's comments there.
