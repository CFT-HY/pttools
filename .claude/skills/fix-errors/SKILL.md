---
name: fix-errors
description: Run the linters, type checks and unit tests of PTtools and fix the errors they find, iterating until all of them pass. First the fast lint (`./lint.sh --fast`), then the full lint (`./lint.sh`) and the unit tests (`uv run pytest`) in parallel. Use when the user asks to fix the errors, lint errors, type errors or failing tests, or to make the checks or the CI pass.
---

# Fix the errors

Follow these steps in order. When a step says **stop**, report the reason to the user and end the skill
without doing any of the later steps.

Redirect the output of the checks to a log file in the scratchpad directory
and inspect it afterwards, as the output can be very long.
Save the exit code of the check itself (e.g. `cmd > log 2>&1; code=$?; tail -n 100 log; exit $code`),
as piping to `tail` would hide a failure.
`./lint.sh` prints `Failed checks:` and the list of the failed checks at the end of its output, if any fail,
and `All checks passed.` otherwise.
For the background checks, print the relevant part of the log in the same command
(e.g. `tail` for `./lint.sh`, and `grep -A60 "short test summary"` and `tail -n 3` for pytest),
so that the output of the background task already contains the results when you are notified.

## How to fix the errors

These rules apply to all the steps below.

- Investigate the cause of each error before fixing it, and fix the cause rather than the symptom.
- The errors are often caused by uncommitted work in progress. Check `git status` and `git diff HEAD`
  to understand what was being changed, and fix the errors in the direction of that change
  instead of reverting it. For example, if a function got a new required parameter and a test or a caller
  was not updated, either update the caller or give the parameter a default value that is consistent
  with the rest of the code (e.g. the same default as in the constructor), and tell the user which one you chose.
  Also check the docstrings of the changed code, as they may have become outdated.
- Follow the instructions of `AGENTS.md`, e.g. the docstring conventions,
  that physics code must be covered by unit tests before editing it,
  and that any changes to the physics must be reported to the user explicitly.
- Do not disable, skip or weaken tests or lint rules to make the checks pass, unless the user agrees.
  This includes `# noqa`, `# pyrefly: ignore`, `pytest.mark.skip`, loosening tolerances
  and updating reference data, unless the error is clearly a false positive or the reference data is clearly outdated.
  In those cases, explain the reason to the user.
- For type errors, follow the conventions in the docstring of `pttools/type_hints.py`:
  prefer targeted `# pyrefly: ignore[<code>]` comments over `typing.cast()` in `@njit` functions,
  as Numba cannot compile `typing.cast()`. In functions that are not JIT-compiled, `typing.cast()` is fine.
- If the type hints (stubs) of a third-party library reject a valid call,
  you may either restructure the call or ignore it with a comment.
  over an ignore comment, and add a short comment explaining why.
- If a check reveals a real bug, do not fix it silently if the fix changes the behavior of the code.
  Report it to the user instead.
- If you cannot fix an error, e.g. because the fix would require a decision by the user,
  or the same error persists after several different attempts, **stop**
  and report the error and what you tried to the user.

## 1. Run the fast lint

1. Run `./lint.sh --fast` (`pyrefly check`, `pyrefly coverage check` and `ruff check`, ~1 min).
2. If it passes, continue to step 2.
3. If it fails, fix the errors and re-run `./lint.sh --fast`. Repeat until it passes.
   When iterating, you may run only the failed checks (e.g. `uv run ruff check`),
   but run the full `./lint.sh --fast` once more at the end, as a fix for one check may break another.

## 2. Run the full lint and the unit tests in parallel

Start both in the background (`run_in_background: true`) in the same response, each with its own log file:

- `./lint.sh` (~2 min alone, ~5 min when run in parallel with the tests). This also runs the documentation lint `python -m pttools.docs.lint`,
  which prints the Sphinx errors and warnings and the LaTeX errors, and saves the full Sphinx output to `./logs/sphinx_TIMESTAMP.log`.
- `uv run pytest` (~10 min on a fast machine, up to 20 min). Do not disable the addopts of `pyproject.toml` with `-o addopts=""`,
  as this would drop `--dist=loadgroup`.

You will be notified when each of them finishes. Do not poll with short sleeps.
Handle each one as soon as it finishes, without waiting for the other.

### 2a. Full lint

If `./lint.sh` fails, fix the errors and re-run it. Repeat until it passes.
When iterating on the documentation errors, you may run only `uv run python -m pttools.docs.lint`,
but run the full `./lint.sh` once more at the end.

### 2b. Unit tests

When all the tests pass, the log has no `short test summary info` section, and the last line is
e.g. `772 passed, 5 xfailed in 590.71s`. The `xfailed` tests are expected failures and need no fixing,
but `xpassed` tests should be reported to the user.

If `uv run pytest` fails, collect the failed tests from the `short test summary info` at the end of the log.
Then fix the errors, and re-run only the failed tests until they pass, e.g.
`uv run pytest tests/test_foo.py::test_bar tests/test_baz.py` or `uv run pytest --last-failed`.
Note that `--last-failed` uses the results of the most recent run, so after a partial re-run
it only includes the tests that failed in that run.

Once all the failed tests pass, check whether the fixes affect code that is run by other tests as well,
e.g. a fix in a library module rather than in a test file.
If so, run all the tests that run that code (e.g. the test files of that module and the tests that use it,
found with `grep`), and fix any new failures in the same way.
If the changed code is used widely (e.g. `pttools/bubble`, `pttools/models` or `pttools/speedup`)
or you cannot tell which tests run it, re-run the full `uv run pytest` instead.

### Interactions between the two

The fixes of one check may break the other, since both run on the same working tree.

- If a fix for the full lint changes code that is run by the tests (i.e. more than comments or docstrings),
  run the tests that run that code once the lint passes, as in step 2b.
- If a fix for the tests changes docstrings or files in `./docs/`, or any code that the type checkers check,
  re-run `./lint.sh` once the tests pass.
- Avoid editing files that the tests are importing while the full `uv run pytest` is still running,
  e.g. by preparing the fixes for the lint after the tests have finished importing the modules (a few minutes).
  If you edit them during a run, the results of that run may not reflect the final code,
  so re-run the affected tests afterwards.

## 3. Final check

If you made any changes in step 2 (not counting step 1, whose changes the checks of step 2 already cover),
run `./lint.sh --fast` once more with the final version of the code,
since the full lint and the tests were run in parallel and their fixes may have affected the fast checks.
If it fails, go back to step 1.

## 4. Report

Summarize to the user:
- the errors found by each check and how they were fixed,
- any changes to the physics, explicitly and exactly what was changed and why,
- any bugs that were found but not fixed, and any errors that could not be fixed,
- the final results of all the checks, including which tests were re-run after the last fixes.

Do not commit or push the changes unless the user asks you to.
