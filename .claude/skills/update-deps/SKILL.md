---
name: update-deps
description: Update the locked dependencies of PTtools with uv, after verifying that the working tree is clean and the latest commit has passed CI, and then fix any issues found by the linters, the unit tests and the documentation build. Use when the user asks to update, upgrade or bump the dependencies or uv.lock.
---

# Update the dependencies

Follow these steps in order. When a step says **stop**, report the reason to the user and end the skill
without doing any of the later steps.

## 1. Check for uncommitted changes

Run `git status --porcelain`.
If it prints anything (modified, staged or untracked files), **stop**
and list the uncommitted changes to the user.

## 2. Check that the latest commit has passed CI

The CI workflow is `.github/workflows/main.yml` with the name `CI`, and it runs on every push.
The repository is `CFT-HY/pttools` (verify with `git remote get-url origin`).

1. Get the latest commit with `git rev-parse HEAD` and the current branch with `git branch --show-current`.
2. Find the CI workflow runs of that commit.
   - If the GitHub plugin (`mcp__github__*` tools) provides a tool for listing GitHub Actions workflow runs
     (e.g. `actions_list` or `list_workflow_runs`), use it. Load it with ToolSearch if it is deferred.
   - Otherwise, fall back to the GitHub CLI:
     `gh run list --workflow CI --commit <sha> --json databaseId,status,conclusion,url,createdAt`
   - If neither works, fall back to the REST API:
     `gh api "repos/CFT-HY/pttools/actions/workflows/main.yml/runs?head_sha=<sha>"`
3. Act on the most recent run:
   - `completed` with conclusion `success`: continue to step 3.
   - `completed` with any other conclusion (`failure`, `cancelled`, `timed_out`, etc.):
     **stop** and give the user the URL of the run and the names of the failed jobs
     (`gh run view <id> --json jobs`).
   - `queued` or `in_progress`: wait for it to finish (see below) and then act on its conclusion.
   - No run exists: the commit has most likely not been pushed. Check with `git status -sb`
     whether the branch is ahead of its upstream. Ask the user for permission to run `git push`
     (use AskUserQuestion). If the user declines, **stop**.
     If the user agrees, run `git push` (or `git push -u origin <branch>` if there is no upstream),
     wait for the CI run of the commit to appear and to finish, and then act on its conclusion.
     If the push does not create a CI run within a few minutes, **stop** and tell the user.

To wait for a run, use `gh run watch <id> --exit-status` in the background (`run_in_background: true`),
as the CI can take several minutes. Do not poll with short sleeps.

## 3. Update the dependencies

1. Run `uv lock --upgrade` to upgrade the locked versions in `uv.lock` to the latest versions
   allowed by `pyproject.toml`.
2. Run `uv sync --all-extras` to install them into `.venv`.
3. Show the user a summary of the version changes, e.g. from `git diff uv.lock`
   (`uv lock --upgrade` also prints them).
   If nothing changed, tell the user that the dependencies are already up to date and **stop**.

Do not loosen or remove version constraints in `pyproject.toml` to get newer versions.
If a constraint prevents an upgrade that seems important, mention it to the user instead.

## 4. Run the checks and fix the issues

Run these checks one at a time, in this order:

1. `./lint.sh` (about 2 min)
2. `uv run pytest` (up to 20 min, run in the background)
3. `make -C docs all` (up to 35 min, run in the background, as it runs the examples)

For each check:
- If it passes, move on to the next check.
- If it fails, investigate the cause, fix it, and re-run the check until it passes before moving on.
  - Prefer fixing the code of PTtools to work with the new dependency versions.
    If a failure is caused by a bug or regression in a dependency, exclude the broken version in `pyproject.toml`
    (e.g. `"package >= X, != Y"`) with a comment linking to the upstream issue, re-run `uv lock` and `uv sync --all-extras`,
    and tell the user.
  - Follow the instructions of `AGENTS.md`, e.g. that physics code must be covered by unit tests before editing it,
    and that any changes to the physics must be reported to the user explicitly.
  - Do not disable, skip or weaken tests or lint rules to make the checks pass, unless the user agrees.
  - If you cannot fix a failure, **stop** and report the failure and what you tried to the user.

If you made any changes to the files other than `uv.lock` during this step,
re-run all three checks at the end with the final version of the code, and ensure that all of them pass.
If a check fails, fix it and repeat the full set of checks.

## 5. Report

Summarize to the user:
- the upgraded packages and their old and new versions,
- the issues found by each check and how they were fixed,
- the final results of all three checks.

Do not commit or push the changes unless the user asks you to.
