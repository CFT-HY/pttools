"""Nightly run: unit tests, documentation build (which runs the examples), wheel build and archiving of the results.

The nightly run is run for multiple repositories in sequence, so that they don't compete for resources.
By default, these are PTtools and PTPlot, if it exists in ``../PTPlot`` relative to the PTtools repository.
The repositories can be selected with ``--repo``.
The outputs of each repository are saved in that repository.

The script is run with ``uv run`` in the virtualenv of PTtools,
so that it can use the utilities of PTtools, e.g. those of :mod:`pttools.utils`:
``uv run --project /path/to/pttools python -m pttools.docs.nightly [arguments]``, see :func:`command`.
The steps of the nightly run of each repository are run in the virtualenv of that repository with ``uv run``,
without the virtualenv of PTtools, see :func:`pttools.utils.uv.env_without_venv`.

Usage:

- ``--status`` or no arguments prints the cron schedule of the nightly run
  and whether a run is currently in progress in each repository.
- ``--run`` runs the nightly run for each repository. This is what cron runs.
- ``--stop`` stops the nightly run in progress and its child processes.
- ``--skip`` skips the next nightly run of each repository.

It is documented in ``./docs/dev.rst``.
"""

import argparse
import collections.abc
import dataclasses
import datetime
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import types
from typing import TextIO, TypedDict, cast

from pttools.logging import append_log, log_console, redirect_output
from pttools.utils.archive import TAR_XZ_SUFFIX, archive, optimize_pngs
from pttools.utils.cron import CronSchedule, read_crontab_entries
from pttools.utils.files import tail
from pttools.utils.git import git
from pttools.utils.lock import acquire_lock, read_lock, write_lock_info
from pttools.utils.process import ProcessInfo, find_descendants, format_process, read_cmdline, read_process, stop
from pttools.utils.repo import Repo
from pttools.utils.time import format_duration, format_time
from pttools.utils.uv import env_without_venv, find_uv

#: The name of this module, with which it is run with ``python -m``
MODULE: str = "pttools.docs.nightly"
SCRIPT_PATH: Path = Path(__file__).resolve()
#: The PTtools repository, in which this script is
SCRIPT_REPO_DIR: Path = SCRIPT_PATH.parent.parent.parent
#: The other repositories, for which the nightly run is run by default if they exist
OPTIONAL_REPO_DIRS: tuple[Path, ...] = (SCRIPT_REPO_DIR.parent / "PTPlot",)
#: The lock that is held while the nightly runs of the repositories are run in sequence,
#: so that two sequences don't run simultaneously and compete for resources
SEQUENCE_LOCK_FILE: Path = SCRIPT_REPO_DIR / "nightly_sequence.lock"
#: The make target of the documentation build.
#: This is like "all", but a failed link check does not stop the build,
#: as it is often caused by a temporarily unreachable server.
DOCS_TARGET: str = "all-nofail-links"
#: The wildcards of the filenames that are left out of the archive, see :mod:`fnmatch`.
#: The checksum files ``*.h5.sha256`` of the HDF5 files are still included.
ARCHIVE_EXCLUDE: tuple[str, ...] = ("*.h5",)
#: The filename of a log of a nightly run: ``nightly_YYYY-MM-DD.log``,
#: or ``nightly_YYYY-MM-DD_2.log``, ``nightly_YYYY-MM-DD_3.log`` etc. for the later runs of the same day
LOG_FILE_PATTERN: re.Pattern[str] = re.compile(r"nightly_(\d{4}-\d{2}-\d{2})(?:_(\d+))?\.log")
#: The regular expression by which the crontab entry of the nightly run is found.
#: It matches both the module name and the path of the script.
CRONTAB_PATTERN: str = r"pttools[./]docs[./]nightly"
#: The default maximum age of the HEAD commit for the run to be started, in seconds.
#: This can be overridden with the ``MAX_COMMIT_AGE`` environment variable.
DEFAULT_MAX_COMMIT_AGE: int = 86400


class LockInfo(TypedDict):
    """Information on a nightly run, which is stored in the lock file."""

    pid: int
    started: str
    head: str
    log_file: str


@dataclasses.dataclass(frozen=True, slots=True)
class NightlyRepo(Repo):
    """A repository for which the nightly run is run, and the paths of the outputs of the nightly run."""

    @property
    def archive_dir(self) -> Path:
        """The directory of the outputs and the archives of the nightly runs."""
        return self.docs_dir / "nightly"

    @property
    def dist_dir(self) -> Path:
        """The output directory of the wheel build of the nightly run."""
        return self.archive_dir / "dist"

    @property
    def docs_build_dir(self) -> Path:
        """The Sphinx build directory of the nightly run, which is separate from the default ``./docs/_build``."""
        return self.archive_dir / "_build"

    @property
    def fig_parent_dir(self) -> Path:
        """The parent directory of the figure directories of the nightly runs, ``fig_YYYY-MM-DD``."""
        return self.archive_dir / "fig"

    @property
    def lock_file(self) -> Path:
        """The lock file of the nightly run of the repository."""
        return self.path / "nightly.lock"

    @property
    def skip_file(self) -> Path:
        """The marker file, whose existence makes the next run to be skipped. It contains the time of the request."""
        return self.path / "nightly.skip"


def command(*args: str, uv: str = "uv") -> str:
    """Get the command with which this script is run with uv in the virtualenv of PTtools.

    :param args: the arguments of the script, e.g. ``--run``
    :param uv: the uv command, e.g. its full path for cron, which does not have it in its PATH
    :return: the command, e.g.
        ``uv run --frozen --quiet --project /path/to/pttools python -m pttools.docs.nightly --run``
    """
    return shlex.join(
        [uv, "run", "--frozen", "--quiet", "--project", str(SCRIPT_REPO_DIR), "python", "-m", MODULE, *args]
    )


def default_repos() -> list[NightlyRepo]:
    """Get the repositories for which the nightly run is run by default.

    :return: PTtools and those of :data:`OPTIONAL_REPO_DIRS` that exist
    """
    optional = [NightlyRepo(path.resolve()) for path in OPTIONAL_REPO_DIRS if path.is_dir()]
    return [NightlyRepo(SCRIPT_REPO_DIR), *optional]


def find_run_process(pid: int | None) -> ProcessInfo | None:
    """Find the process of the nightly run that holds a lock.

    :param pid: the process ID read from the lock file
    :return: the process, or ``None`` if the process is not running a nightly run
    """
    if pid is None:
        return None
    cmdline = read_cmdline(pid)
    if "--run" not in cmdline or not (MODULE in cmdline or any(arg.endswith("nightly.py") for arg in cmdline)):
        return None
    return read_process(pid)


def read_lock_file(repo: NightlyRepo) -> tuple[bool, LockInfo | None]:
    """Check whether the lock of the nightly run of a repository is held, and read the information from it.

    :param repo: the repository
    :return: whether the lock is held, and the information on the current or latest run, if available
    """
    locked, info = read_lock(repo.lock_file)
    return locked, cast(LockInfo | None, info)


def log_files(repo: NightlyRepo) -> list[Path]:
    """The log files of the nightly runs of a repository, in the order of the runs.

    :param repo: the repository
    :return: the paths of the log files that match :data:`LOG_FILE_PATTERN`, from the oldest to the latest
    """
    logs = []
    for path in repo.log_dir.glob("nightly_*.log"):
        match = LOG_FILE_PATTERN.fullmatch(path.name)
        if match is not None:
            logs.append(((match[1], int(match[2] or 1)), path))
    return [path for _, path in sorted(logs)]


# -- Status ------------------------------------------------------------------


def next_scheduled_run(now: datetime.datetime) -> datetime.datetime | None:
    """Get the time of the next scheduled nightly run from the crontab.

    :param now: the current local time
    :return: the time of the next run, or ``None`` if no run is scheduled or the schedule could not be parsed
    """
    next_runs = []
    for entry in read_crontab_entries(CRONTAB_PATTERN) or []:
        if entry.lstrip().startswith("#") or "--run" not in entry.split():
            continue
        try:
            next_run = CronSchedule.parse(entry).next_run(now)
        except ValueError:
            continue
        if next_run is not None:
            next_runs.append(next_run)
    return min(next_runs, default=None)


def read_skip_request(repo: NightlyRepo) -> datetime.datetime | None:
    """Check whether skipping the next run has been requested with ``--skip``.

    :param repo: the repository
    :return: the time of the request, or ``None`` if skipping has not been requested
    """
    try:
        content = repo.skip_file.read_text().strip()
    except FileNotFoundError:
        return None
    try:
        return datetime.datetime.fromisoformat(content)
    except ValueError:
        # The file has been created by other means, e.g. with touch.
        return datetime.datetime.fromtimestamp(repo.skip_file.stat().st_mtime)


def print_crontab() -> None:
    """Print the crontab entries of the nightly run and their schedules."""
    print(f'Crontab entry (crontab -l | grep -E "{CRONTAB_PATTERN}"):')
    entries = read_crontab_entries(CRONTAB_PATTERN)
    if entries is None:
        print("  (crontab was not found)")
        return
    if not entries:
        print("  (none, the nightly run is not scheduled)")
        return
    now = datetime.datetime.now()
    for entry in entries:
        print(f"  {entry}")
        if entry.lstrip().startswith("#"):
            print("  (commented out)")
            continue
        if "--run" not in entry.split():
            print("  WARNING: the entry does not have the --run argument, so it only prints the status.")
        if "uv" not in entry and not any(arg.endswith("/uv") for arg in entry.split()):
            print(
                f"  WARNING: the entry does not run the script with uv. Use: {command('--run', uv=find_uv() or 'uv')}"
            )
        try:
            schedule = CronSchedule.parse(entry)
        except ValueError:
            print("  (the schedule could not be parsed)")
            continue
        description = schedule.describe()
        if description is None:
            fields = " ".join(entry.split()[:5])
            description = f"'{fields}' (minute hour day-of-month month day-of-week, see crontab(5))"
        print(f"  Schedule: {description}, in the local time zone ({now.astimezone().strftime('%Z')})")
        next_run = schedule.next_run(now)
        if next_run is not None:
            print(
                f"  Next run: {format_time(next_run)}, "
                f"in {format_duration(next_run - now, seconds=False)}"
            )


def status(repos: collections.abc.Sequence[NightlyRepo]) -> None:
    """Print the status of the nightly runs.

    :param repos: the repositories
    """
    now = datetime.datetime.now()
    print(f"The nightly run is run with: {command('--run', uv=find_uv() or 'uv')}")
    print(f"Current time: {format_time(now)}")
    print()
    print_crontab()
    print(f"Repositories, in the order of the runs: {', '.join(repo.name for repo in repos)}")
    for repo in repos:
        print()
        print(f"=== {repo.name} ({repo.path}) ===")
        repo_status(repo, now)


def repo_status(repo: NightlyRepo, now: datetime.datetime) -> None:
    """Print the status of the nightly run of a repository.

    :param repo: the repository
    :param now: the current local time
    """
    skip_requested = read_skip_request(repo)
    if skip_requested is None:
        print("Next run will be skipped: NO")
    else:
        print(f"Next run will be skipped: YES, as requested with --skip at {format_time(skip_requested)}")
        print(f"  To cancel the skip, delete {repo.skip_file}")

    locked, info = read_lock_file(repo)
    print(f"Nightly run in progress: {'YES' if locked else 'NO'}")
    if info is not None:
        started = datetime.datetime.fromisoformat(info["started"])
        print(f"  {'Current' if locked else 'Latest'} run started: {format_time(started)}")
        if locked:
            print(f"  Running for: {format_duration(now - started, seconds=False)}")
        print(f"  PID: {info['pid']}, HEAD: {info['head']}")
        print(f"  Log: {info['log_file']}")
    if locked:
        process = find_run_process(info["pid"] if info is not None else None)
        if process is None:
            print("  WARNING: the lock is held, but the process in the lock file is not a nightly run.")
        else:
            print("  Processes:")
            for descendant in [process, *find_descendants([process.pid]).values()]:
                print(f"  {format_process(descendant)}")
        print(f"  The run can be stopped with: {command('--stop')}")

    logs = log_files(repo)
    if not logs:
        print(f"Last lines of the latest log: (no logs in {repo.log_dir})")
        return
    print(f"Last lines of the latest log {logs[-1]}:")
    for line in tail(logs[-1], 5):
        print(f"  {line}")


# -- Nightly run -------------------------------------------------------------


def create_log_file(repo: NightlyRepo, date: datetime.date) -> Path:
    """Create a new log file for a nightly run of a repository.

    The log file is ``nightly_YYYY-MM-DD.log``, or if it already exists,
    ``nightly_YYYY-MM-DD_2.log``, ``nightly_YYYY-MM-DD_3.log`` etc.
    The file is created atomically, so that two runs never get the same log file.

    :param repo: the repository
    :param date: the date of the run
    :return: the path of the created log file
    """
    repo.log_dir.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        log_file = repo.log_dir / f"nightly_{date:%Y-%m-%d}{f'_{i}' if i > 1 else ''}.log"
        try:
            os.close(os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        except FileExistsError:
            i += 1
        else:
            return log_file


def latest_log_file(repo: NightlyRepo, date: datetime.date) -> Path:
    """The latest log file of a repository of a date, to which messages of the runs that were not started are appended.

    :param repo: the repository
    :param date: the date
    :return: the path of the latest existing log file of the date, or of a new log file if there is none
    """
    logs = [log_file for log_file in log_files(repo) if log_file.name.startswith(f"nightly_{date:%Y-%m-%d}")]
    if logs:
        return logs[-1]
    return create_log_file(repo, date)


def nightly_run(repo: NightlyRepo, uv: str | None, lock_fd: int, log_file: Path, started: datetime.datetime) -> int:
    """Run the unit tests, build the docs and the wheel, optimize the PNG files and archive the results of a repo.

    The completion of the run is logged before archiving, so that the archived log includes it.
    The creation of the archive is then logged only to the log file.

    :param repo: the repository
    :param uv: the path of uv from :func:`find_uv`
    :param lock_fd: the file descriptor of the lock file of the repository
    :param log_file: the path of the log file
    :param started: the start time of the run
    :return: the exit code
    """
    head = git(repo.path, "rev-parse", "--short", "HEAD")
    write_lock_info(
        lock_fd, {"pid": os.getpid(), "started": started.isoformat(), "head": head, "log_file": str(log_file)}
    )

    skip_requested = read_skip_request(repo)
    if skip_requested is not None:
        repo.skip_file.unlink(missing_ok=True)
        log_console(
            f"Skipping the nightly run, as requested with --skip at {skip_requested.isoformat(timespec='seconds')}."
        )
        return 0

    # Run only if the HEAD commit is recent, so that unchanged code is not rebuilt every night.
    max_commit_age = int(os.environ.get("MAX_COMMIT_AGE", DEFAULT_MAX_COMMIT_AGE))
    commit_age = time.time() - int(git(repo.path, "log", "-1", "--format=%ct", "HEAD"))
    if commit_age > max_commit_age:
        log_console(f"HEAD ({head}) is {int(commit_age // 3600)} h old. Skipping the nightly run.")
        return 0

    log_console(f"Starting the nightly run of {repo.path} at HEAD {head}.")

    # The build directory is cleaned also by the make target,
    # but cleaning it already here ensures that no output of a previous run is left there if the run fails.
    for directory in (repo.docs_build_dir, repo.dist_dir):
        if directory.exists():
            shutil.rmtree(directory)
            log_console(f"Cleaned {directory}.")
    fig_dir = repo.fig_parent_dir / f"fig_{started:%Y-%m-%d}"
    # The examples save their figures in this directory.
    env = {**env_without_venv(), repo.fig_dir_env: str(fig_dir)}
    log_console(f"Saving the figures to {fig_dir} (${repo.fig_dir_env}).")

    if uv is None:
        log_console(f"uv was not found in PATH: {os.environ['PATH']}")
        return 1
    uv_version = subprocess.run([uv, "--version"], stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    log_console(f"Using {uv} ({uv_version}), and running the script with {sys.executable}.")

    # Run in the virtualenv that uv manages.
    # The --frozen ensures that the locked dependency versions are used as they are, without updating uv.lock.
    uv_run = [uv, "run", "--frozen", "--project", str(repo.path)]

    subprocess.run([*uv_run, "pytest"], cwd=repo.path, env=env, check=True)
    log_console("Unit tests finished.")

    # A variable given on the command line of make overrides the one in the Makefile,
    # and it is passed on to the recursive make calls.
    subprocess.run(
        [*uv_run, "make", "-C", str(repo.docs_dir), DOCS_TARGET, f"BUILDDIR={repo.docs_build_dir}"],
        cwd=repo.path,
        env=env,
        check=True,
    )
    log_console("Documentation build finished.")

    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(repo.dist_dir), str(repo.path)], cwd=repo.path, env=env, check=True
    )
    log_console(f"Wheel build finished: {', '.join(wheel.name for wheel in sorted(repo.dist_dir.glob('*.whl')))}.")

    optimize_pngs([repo.docs_build_dir, fig_dir])

    # This is logged before archiving, so that the archived log includes it.
    log_console("Nightly run finished. Archiving the results.")
    # A copy of the log is archived, so that the archive is not affected by anything written to the log meanwhile.
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_copy = Path(tmp_dir) / log_file.name
        shutil.copy2(log_file, log_copy)
        archive(
            repo.archive_dir / f"{repo.name.lower()}_nightly_{started:%Y-%m-%d}{TAR_XZ_SUFFIX}",
            {
                repo.docs_build_dir: repo.docs_build_dir.name,
                fig_dir: "fig",
                repo.dist_dir: repo.dist_dir.name,
                log_copy: log_file.name,
            },
            exclude=ARCHIVE_EXCLUDE,
        )
    return 0


class StopRequested(BaseException):
    """Raised in the process of the nightly run when it receives SIGTERM.

    This inherits from :class:`BaseException` so that it is not caught by ``except Exception``.
    When :func:`subprocess.run` is interrupted by an exception, it kills its child process.
    """


def raise_stop_requested(_signum: int, _frame: types.FrameType | None) -> None:
    """Signal handler for SIGTERM.

    :param _signum: the signal number
    :param _frame: the current stack frame
    :raises StopRequested: always
    """
    raise StopRequested


def run_repo(repo: NightlyRepo, uv: str | None, original_stdout: TextIO, original_stderr: TextIO) -> int:
    """Run the nightly run of a repository, unless another run of it is already in progress.

    All output goes to the log file of the repository.
    Only if the run fails, a short message is printed to the original stderr,
    so that cron sends an email only of the failed runs.

    :param repo: the repository
    :param uv: the path of uv from :func:`find_uv`
    :param original_stdout: the stdout of the script before the redirection to the log file
    :param original_stderr: the stderr of the script before the redirection to the log file
    :return: the exit code
    :raises StopRequested: if the run was stopped with SIGTERM,
        so that the runs of the next repositories are not started
    """
    started = datetime.datetime.now()
    lock_fd = acquire_lock(repo.lock_file)
    if lock_fd is None:
        _, info = read_lock_file(repo)
        pid = info["pid"] if info is not None else "unknown"
        append_log(
            latest_log_file(repo, started.date()),
            f"Another nightly run (PID {pid}) of {repo.name} is in progress. Skipping {repo.name}.",
        )
        return 1

    log_file = create_log_file(repo, started.date())
    if original_stdout.isatty():
        print(f"Logging the nightly run of {repo.name} to {log_file}", file=original_stdout, flush=True)
    redirect_output(log_file)

    exit_code = 1
    stopped = False
    try:
        exit_code = nightly_run(repo, uv, lock_fd, log_file, started)
    except subprocess.CalledProcessError as error:
        log_console(f"FAILED: {' '.join(map(str, error.cmd))} returned exit code {error.returncode}.")
        exit_code = error.returncode
    except StopRequested:
        log_console("Stopped by SIGTERM, e.g. by --stop.")
        stopped = True
        raise
    except Exception:  # noqa: BLE001
        # The runs of the other repositories are continued after an unexpected error.
        log_console("FAILED with the following exception.")
        traceback.print_exc()
        sys.stderr.flush()
    except BaseException:
        log_console("FAILED with the following exception.")
        raise
    finally:
        os.close(lock_fd)
        # A deliberately stopped run is not reported as a failure.
        if exit_code and not stopped:
            print(
                f"The nightly run of {repo.name} failed with exit code {exit_code}. See the log {log_file}",
                file=original_stderr,
                flush=True,
            )
    return exit_code


def run(repos: collections.abc.Sequence[NightlyRepo]) -> int:
    """Run the nightly runs of the repositories in sequence, unless another sequence is already in progress.

    The runs are continued for the next repositories even if the run of a repository fails,
    but not if the run is stopped with ``--stop``.

    :param repos: the repositories
    :return: 0 if all the runs succeeded, and otherwise the exit code of the first failed run
    """
    sequence_fd = acquire_lock(SEQUENCE_LOCK_FILE)
    if sequence_fd is None:
        _, info = read_lock(SEQUENCE_LOCK_FILE)
        pid = info.get("pid", "unknown") if info is not None else "unknown"
        message = f"Another nightly run (PID {pid}) is in progress. Exiting."
        if sys.stdout.isatty():
            print(message)
        for repo in repos:
            append_log(latest_log_file(repo, datetime.date.today()), message)
        return 1

    signal.signal(signal.SIGTERM, raise_stop_requested)
    exit_code = 0
    try:
        write_lock_info(sequence_fd, {"pid": os.getpid(), "started": datetime.datetime.now().isoformat()})
        uv = find_uv()
        with (
            os.fdopen(os.dup(sys.stdout.fileno()), "w") as original_stdout,
            os.fdopen(os.dup(sys.stderr.fileno()), "w") as original_stderr,
        ):
            for repo in repos:
                repo_exit_code = run_repo(repo, uv, original_stdout, original_stderr)
                exit_code = exit_code or repo_exit_code
    except StopRequested:
        # The stop has already been logged by run_repo, unless it happened between the runs.
        return 128 + signal.SIGTERM
    finally:
        os.close(sequence_fd)
    return exit_code


# -- Stopping ----------------------------------------------------------------


def stop_run(repos: collections.abc.Sequence[NightlyRepo], timeout: float) -> int:
    """Stop the nightly run that is in progress and all its child processes.

    The runs of the next repositories in the sequence are not started.
    First, SIGTERM is sent to the nightly run process, which then logs that it has been stopped and exits,
    and to all its descendant processes.
    The processes that are still running after the timeout are killed with SIGKILL.
    The processes are identified by their PIDs and start times,
    so that the processes that have been reparented to init after the exit of their parents are also stopped,
    and that no unrelated processes are signalled, even if their PIDs have been reused.

    :param repos: the repositories, to the logs of whose current runs the stop is logged
    :param timeout: the time to wait after SIGTERM before sending SIGKILL, in seconds
    :return: the exit code
    """
    locked, sequence_info = read_lock(SEQUENCE_LOCK_FILE)
    if not locked:
        print("No nightly run is in progress.")
        return 0
    pid = sequence_info.get("pid") if sequence_info is not None else None
    process = find_run_process(pid if isinstance(pid, int) else None)
    if process is None:
        print(
            f"The lock {SEQUENCE_LOCK_FILE} is held, but the process in the lock file is not a nightly run. "
            "Please check the processes manually."
        )
        return 1
    log_files = []
    for repo in repos:
        repo_locked, info = read_lock_file(repo)
        if repo_locked and info is not None and info["pid"] == process.pid:
            log_files.append(Path(info["log_file"]))

    targets = {process.pid: process, **find_descendants([process.pid])}
    print(f"Stopping the nightly run (PID {process.pid}) and its child processes:")
    for target in targets.values():
        print(format_process(target))
    # The nightly run process is signalled first, as it is the first in the dict,
    # so that it does not start new steps when its child processes exit.
    if stop(targets, timeout):
        return 1

    for log_file in log_files:
        if log_file.exists():
            append_log(
                log_file, f"The nightly run and its {len(targets) - 1} child processes were stopped with --stop."
            )
    print(f"The nightly run and its {len(targets) - 1} child processes were stopped.")
    return 0


# -- Skipping ----------------------------------------------------------------


def skip(repos: collections.abc.Sequence[NightlyRepo]) -> int:
    """Request the next nightly run of each repository to be skipped.

    The request is consumed by the next ``--run`` that gets the lock, whether it is started by cron or manually.

    :param repos: the repositories
    :return: the exit code
    """
    now = datetime.datetime.now()
    for repo in repos:
        skip_requested = read_skip_request(repo)
        if skip_requested is None:
            repo.skip_file.write_text(now.isoformat() + "\n")
            print(f"The next nightly run of {repo.name} will be skipped.")
        else:
            print(
                f"Skipping the next nightly run of {repo.name} has already been requested "
                f"at {format_time(skip_requested)}."
            )
        if read_lock_file(repo)[0]:
            print(
                f"The nightly run of {repo.name} that is currently in progress is not affected. "
                "To stop it, use --stop."
            )
    next_run = next_scheduled_run(now)
    if next_run is None:
        print("No nightly run is scheduled in the crontab.")
    else:
        print(
            f"Next scheduled run: {format_time(next_run)}, "
            f"in {format_duration(next_run - now, seconds=False)}"
        )
    for repo in repos:
        print(f"To cancel the skip of {repo.name}, delete {repo.skip_file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run, stop or skip the nightly run with ``--run``, ``--stop`` or ``--skip``, or print its status.

    The status is printed with ``--status`` or without arguments.

    :param argv: the command-line arguments
    :return: the exit code
    """
    parser = argparse.ArgumentParser(
        description="With --status or without arguments, print the cron schedule of the nightly run "
        "and whether a run is currently in progress in each repository."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="print the status (the default without arguments)")
    group.add_argument("--run", action="store_true", help="run the nightly run (this is what cron runs)")
    group.add_argument("--stop", action="store_true", help="stop the nightly run in progress and its child processes")
    group.add_argument("--skip", action="store_true", help="skip the next nightly run of each repository")
    parser.add_argument(
        "--repo",
        type=Path,
        action="append",
        dest="repos",
        help="a repository for which the nightly run is run, in the given order, instead of the default ones: "
        f"{', '.join(str(repo.path) for repo in default_repos())}. Can be given multiple times.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10,
        help="with --stop, the time to wait after SIGTERM before sending SIGKILL, in seconds (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    repos = [NightlyRepo(path.resolve()) for path in args.repos] if args.repos else default_repos()
    for repo in repos:
        if not repo.path.is_dir():
            parser.error(f"The repository {repo.path} does not exist.")
    if not (SCRIPT_REPO_DIR / ".git").exists():
        parser.error(
            f"The nightly run has to be run from the PTtools repository, but {SCRIPT_PATH} is not in one. "
            f"Run it with: uv run --project /path/to/pttools python -m {MODULE} [arguments]"
        )
    if args.run:
        return run(repos)
    if args.stop:
        return stop_run(repos, args.timeout)
    if args.skip:
        return skip(repos)
    status(repos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
