---
name: update-deps
description: Update the locked dependencies of PTtools with uv and the CUDA base image of the Dockerfile, after verifying that the working tree is clean and the latest commit has passed CI, and then fix any issues found by the linters, the unit tests, the documentation build and the Docker build. Use when the user asks to update, upgrade or bump the dependencies, uv.lock or the Docker base image.
---

# Update the dependencies

Follow these steps in order. When a step says **stop**, report the reason to the user and end the skill
without doing any of the later steps.

## 1. Check for uncommitted changes

Run `git status --porcelain -- . ':!.claude'`.
If it prints anything (modified, staged or untracked files), **stop**
and list the uncommitted changes to the user.

## 2. Check that the latest commit has passed CI

The CI workflow is `.github/workflows/main.yml` with the name `CI`, and it runs on every push.
It includes all the checks: the lint, the type checks, the build, the tests on Linux, Windows and macOS,
and the documentation. The jobs from the reusable workflows are named e.g. `test-windows / test`.
The repository is `CFT-HY/pttools` (verify with `git remote get-url origin`).

1. Get the latest commit with `git rev-parse HEAD` and the current branch with `git branch --show-current`.
2. Find the CI workflow runs of that commit.
   - If the GitHub plugin (`mcp__github__*` tools) provides a tool for listing GitHub Actions workflow runs
     (e.g. `actions_list` or `list_workflow_runs`), use it. Load it with ToolSearch if it is deferred.
     (As of 2026-09, the plugin does not have the Actions toolset enabled, so the fallback is usually needed.)
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

## 3. Update the Python dependencies

1. Run `uv lock --upgrade` to upgrade the locked versions in `uv.lock` to the latest versions
   allowed by `pyproject.toml`.
2. Run `uv sync --all-extras` to install them into `.venv`.
3. Show the user a summary of the version changes, e.g. from `git diff uv.lock`
   (`uv lock --upgrade` also prints them).

Do not loosen or remove version constraints in `pyproject.toml` to get newer versions.
If a constraint prevents an upgrade that seems important, mention it to the user instead.

## 4. Update the base image of the Docker image

The base image is set in `./Dockerfile` by `ARG CUDA_IMAGE="nvidia/cuda:<CUDA version>-base-ubuntu<Ubuntu version>"`,
e.g. `nvidia/cuda:13.4.2-base-ubuntu26.04`. Both version numbers can change, but `-base-ubuntu` stays the same.

1. Run `uv run python -m pttools.utils.cuda_image`.
   It reads the current tag from the Dockerfile, fetches the tags from Docker Hub, and prints
   the latest CUDA version for each Ubuntu LTS release that has images for both `linux/amd64` and `linux/arm64`
   (the platforms of `.github/actions/deploy-docker/action.yml`), and then the current and the latest tag,
   and which version numbers would change. Add `--all` to list every tag instead of only the latest ones.
   If the script fails, e.g. due to a change in the Docker Hub API, fix it, or fall back to
   `curl -s "https://hub.docker.com/v2/repositories/nvidia/cuda/tags?page_size=100&name=-base-ubuntu&ordering=last_updated" | jq -r '.results[].name'`.
2. If the script prints "The base image is up to date", continue to step 5.
3. If the CUDA major version would change (e.g. 13 -> 14), ask the user before updating (use AskUserQuestion),
   since a new CUDA major version requires a newer NVIDIA driver on the host and may drop support for older GPUs.
   If the user declines, use the latest tag with the current CUDA major version
   and the current Ubuntu version instead (see the `--all` listing).
4. If the Ubuntu version would change, check the new release before updating
   (set `IMAGE` to the new image, e.g. `nvidia/cuda:13.4.2-base-ubuntu28.04`):
   ```bash
   docker run --rm "$IMAGE" bash -c "apt-get update -qq && apt-cache policy python3 build-essential cmake gfortran patchelf python3-dev libgfortran5 libgomp1"
   ```
   - The image uses the system Python of Ubuntu (`UV_PYTHON_DOWNLOADS=never`), so its version (`Candidate:` of `python3`)
     must match `.python-version` and satisfy `requires-python` of `pyproject.toml`.
     Otherwise `uv sync` in the Docker build cannot find a suitable interpreter.
     If it does not match, keep the current Ubuntu version and use the latest CUDA version for it,
     and tell the user that a newer Ubuntu release is available, but requires a different Python version.
   - All the apt packages that are installed in the Dockerfile must have a candidate (`Candidate:` is not `(none)`).
     If e.g. `libgfortran5` has been renamed, update the package name in the Dockerfile.
   - Update the comment `# Ubuntu XX.YY includes Python 3.Z.` above `ARG CUDA_IMAGE` accordingly.
   - The CI job `test-arm` in `.github/workflows/main.yml` runs on a specific Ubuntu version (e.g. `ubuntu-26.04-arm`).
     Do not change it, but mention it to the user.
5. Replace the tag in the `ARG CUDA_IMAGE` line with the chosen tag,
   and re-run the script to verify that it reports the new tag as the current one.

If neither `uv.lock` nor `Dockerfile` changed in steps 3 and 4,
tell the user that the dependencies and the base image are already up to date and **stop**.

## 5. Run the checks and fix the issues

Run these checks one at a time, in this order:

1. `./lint.sh` (~2 min)
2. `uv run pytest` (~20 min, run in the background)
3. `uv run make -C docs all` (up to 35 min, run in the background, as it runs the examples).
4. The Docker build (~5 min, run in the background), as both `uv.lock` and the base image affect it:
   ```bash
   docker build --check . && docker build -t pttools:update-deps . && docker run --rm pttools:update-deps python -c "import numbalsoda; from pttools.bubble import Bubble; from pttools.models import BagModel; from pttools.omgw0 import Spectrum; from pttools.ssm import NucType; bubble = Bubble(BagModel(alpha_n_min=0.01), v_wall=0.5, alpha_n=0.2); print(bubble.kappa, Spectrum(bubble, nuc_type=NucType.EXPONENTIAL, r_star=0.1).omgw0()[0])"
   ```
   The smoke test should print the same numbers as the same command with `uv run python -c "..."` outside Docker.
   This builds the image only for the platform of the local machine.
   The CI also builds it for the other platforms, but only after the tests have passed.
   If Docker is not available, skip this check and tell the user.
   Afterwards, remove the test image with `docker image rm pttools:update-deps`.

Redirect the output of the long checks to a log file in the scratchpad directory
and inspect its tail afterwards, as the output is very long.
Save the exit code of the check itself (e.g. `cmd > log 2>&1; code=$?; tail log; exit $code`),
as piping to `tail` would hide a failure.

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
  - If a new version of `pyrefly` reports new errors, compare with the old version
    (`uv run --with pyrefly==<old> pyrefly check --output-format min-text`)
    to tell apart the errors caused by the type checker from those caused by the upgraded stubs of other packages.
    Follow the conventions in the docstring of `pttools/type_hints.py`:
    prefer targeted `# pyrefly: ignore[<code>]` comments over `typing.cast()` in `@njit` functions,
    as Numba cannot compile `typing.cast()`.
  - If a type checker reveals a real bug, do not fix it silently if the fix changes the behavior of the code.
    Report it to the user instead.
  - If you cannot fix a failure, **stop** and report the failure and what you tried to the user.

If you made any changes to the files other than `uv.lock` and `Dockerfile` during this step,
re-run all the checks at the end with the final version of the code, and ensure that all of them pass.
If a check fails, fix it and repeat the full set of checks.

## 6. Report

Summarize to the user:
- the upgraded packages and their old and new versions,
- the old and new base image of the Dockerfile, or why it was not updated to the latest tag,
- the issues found by each check and how they were fixed,
- the final results of all the checks.

Do not commit or push the changes unless the user asks you to.
