#!/usr/bin/env python3
"""Nightly run: unit tests, documentation build (which runs the examples), wheel build and archiving of the results.

The nightly run is run for multiple repositories in sequence, so that they don't compete for resources.
By default, these are PTtools and PTPlot, if it exists in ``../PTPlot`` relative to the PTtools repository.
The repositories can be selected with ``--repo``.
The outputs of each repository are saved in that repository.

Usage:

- ``pttools/docs/nightly.py`` or ``pttools/docs/nightly.py --status`` prints the cron schedule of the nightly run
  and whether a run is currently in progress in each repository.
- ``pttools/docs/nightly.py --run`` runs the nightly run for each repository. This is what cron runs.
- ``pttools/docs/nightly.py --stop`` stops the nightly run in progress and its child processes.
- ``pttools/docs/nightly.py --skip`` skips the next nightly run of each repository.

The script is run with the Python of the system and not in the virtualenv that uv manages,
and therefore it may only use the Python standard library.
It is documented in ``./docs/dev.rst``.
"""

import argparse
import collections.abc
import dataclasses
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import types
from typing import Any, Self, TextIO, TypedDict, cast

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
#: The 7-Zip wildcards of the files that are left out of the archive.
#: The checksum files ``*.h5.sha256`` of the HDF5 files are still included.
ARCHIVE_EXCLUDE: tuple[str, ...] = ("*.h5",)
#: The filename of a log of a nightly run: ``nightly_YYYY-MM-DD.log``,
#: or ``nightly_YYYY-MM-DD_2.log``, ``nightly_YYYY-MM-DD_3.log`` etc. for the later runs of the same day
LOG_FILE_PATTERN: re.Pattern[str] = re.compile(r"nightly_(\d{4}-\d{2}-\d{2})(?:_(\d+))?\.log")
#: The command with which the nightly run is run
RUN_COMMAND: str = f"{SCRIPT_PATH} --run"
#: The string by which the crontab entry of the nightly run is found
CRONTAB_PATTERN: str = "pttools/docs/nightly.py"
#: The default maximum age of the HEAD commit for the run to be started, in seconds.
#: This can be overridden with the ``MAX_COMMIT_AGE`` environment variable.
DEFAULT_MAX_COMMIT_AGE: int = 86400

#: The cron schedule nicknames, see crontab(5)
CRON_NICKNAMES: dict[str, str] = {
    "@yearly": "0 0 1 1 *",
    "@annually": "0 0 1 1 *",
    "@monthly": "0 0 1 * *",
    "@weekly": "0 0 * * 0",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@hourly": "0 * * * *",
}
WEEKDAY_NAMES: tuple[str, ...] = ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")


class LockInfo(TypedDict):
    """Information on a nightly run, which is stored in the lock file."""

    pid: int
    started: str
    head: str
    log_file: str


@dataclasses.dataclass(frozen=True)
class Repo:
    """A repository for which the nightly run is run, and the paths of its outputs."""

    #: The root directory of the repository
    path: Path

    @property
    def name(self) -> str:
        """The name of the repository directory."""
        return self.path.name

    @property
    def log_dir(self) -> Path:
        """The directory of the logs ``nightly_YYYY-MM-DD.log``, see :data:`LOG_FILE_PATTERN`."""
        return self.path / "logs"

    @property
    def lock_file(self) -> Path:
        """The lock file of the nightly run of the repository."""
        return self.path / "nightly.lock"

    @property
    def skip_file(self) -> Path:
        """The marker file, whose existence makes the next run to be skipped. It contains the time of the request."""
        return self.path / "nightly.skip"

    @property
    def docs_dir(self) -> Path:
        """The documentation directory."""
        return self.path / "docs"

    @property
    def archive_dir(self) -> Path:
        """The directory of the outputs and the archives of the nightly runs."""
        return self.docs_dir / "nightly"

    @property
    def docs_build_dir(self) -> Path:
        """The Sphinx build directory of the nightly run, which is separate from the default ``./docs/_build``."""
        return self.archive_dir / "_build"

    @property
    def dist_dir(self) -> Path:
        """The output directory of the wheel of the nightly run."""
        return self.archive_dir / "dist"

    @property
    def fig_parent_dir(self) -> Path:
        """The parent directory of the figure directories ``fig_YYYY-MM-DD`` of the nightly runs."""
        return self.archive_dir / "fig"

    @property
    def fig_dir_env(self) -> str:
        """The environment variable with which the figure directory of the examples is set.

        It's ``NAME_FIG_DIR``, where ``NAME`` is the name of the repository in upper case,
        e.g. ``PTTOOLS_FIG_DIR`` or ``PTPLOT_FIG_DIR``. See ``examples/utils.py`` of the repository.
        """
        return re.sub(r"\W", "_", self.name).upper() + "_FIG_DIR"


def default_repos() -> list[Repo]:
    """Get the repositories for which the nightly run is run by default.

    :return: PTtools and those of :data:`OPTIONAL_REPO_DIRS` that exist
    """
    return [Repo(SCRIPT_REPO_DIR), *(Repo(path.resolve()) for path in OPTIONAL_REPO_DIRS if path.is_dir())]


# -- Cron schedule -----------------------------------------------------------


def parse_cron_field(field: str, low: int, high: int) -> frozenset[int]:
    """Parse a field of a cron schedule.

    Supports ``*``, numbers, ranges ``a-b``, steps ``/n`` and comma-separated lists of these.
    Names of months and days of the week are not supported.

    :param field: the field, e.g. ``*/15`` or ``1-5``
    :param low: the minimum allowed value
    :param high: the maximum allowed value
    :return: the values that the field matches
    :raises ValueError: if the field is invalid
    """
    values: set[int] = set()
    for part in field.split(","):
        base, _, step_str = part.partition("/")
        step = int(step_str) if step_str else 1
        if step < 1:
            raise ValueError(f"Invalid step in the cron field: {field}")
        if base == "*":
            start, end = low, high
        elif "-" in base:
            start_str, end_str = base.split("-", 1)
            start, end = int(start_str), int(end_str)
        else:
            start = int(base)
            end = high if step_str else start
        if not low <= start <= end <= high:
            raise ValueError(f"Invalid cron field: {field}")
        values.update(range(start, end + 1, step))
    return frozenset(values)


@dataclasses.dataclass(frozen=True)
class CronSchedule:
    """The schedule of a cron job."""

    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    #: The days of the week, where 0 is Sunday
    weekdays: frozenset[int]
    #: Whether the day of the month field is restricted, i.e. does not start with ``*``
    days_restricted: bool
    #: Whether the day of the week field is restricted, i.e. does not start with ``*``
    weekdays_restricted: bool

    @classmethod
    def parse(cls, schedule: str) -> Self:
        """Parse the schedule from the beginning of a crontab line.

        :param schedule: the crontab line, e.g. ``0 2 * * * command``
        :return: the parsed schedule
        :raises ValueError: if the schedule is invalid or not supported
        """
        fields = schedule.split()
        if fields and fields[0].startswith("@"):
            if fields[0] not in CRON_NICKNAMES:
                raise ValueError(f"Unsupported cron schedule: {fields[0]}")
            fields = CRON_NICKNAMES[fields[0]].split()
        if len(fields) < 5:  # noqa: PLR2004
            raise ValueError(f"Invalid cron schedule: {schedule}")
        minute, hour, day, month, weekday = fields[:5]
        return cls(
            minutes=parse_cron_field(minute, 0, 59),
            hours=parse_cron_field(hour, 0, 23),
            days=parse_cron_field(day, 1, 31),
            months=parse_cron_field(month, 1, 12),
            # Both 0 and 7 are Sunday.
            weekdays=frozenset(value % 7 for value in parse_cron_field(weekday, 0, 7)),
            days_restricted=not day.startswith("*"),
            weekdays_restricted=not weekday.startswith("*"),
        )

    def matches_date(self, date: datetime.date) -> bool:
        """Check whether the job is run on the given date.

        If both the day of the month and the day of the week are restricted,
        the job is run when either of them matches, as in cron.

        :param date: the date
        :return: whether the job is run on the date
        """
        if date.month not in self.months:
            return False
        day_match = date.day in self.days
        weekday_match = date.isoweekday() % 7 in self.weekdays
        if self.days_restricted and self.weekdays_restricted:
            return day_match or weekday_match
        return day_match and weekday_match

    def next_run(self, now: datetime.datetime) -> datetime.datetime | None:
        """Get the next time when the job is run.

        :param now: the current local time
        :return: the next run time, or ``None`` if there is none within the next four years
        """
        for offset in range(4 * 366 + 1):
            date = now.date() + datetime.timedelta(days=offset)
            if not self.matches_date(date):
                continue
            for hour in sorted(self.hours):
                for minute in sorted(self.minutes):
                    run_time = datetime.datetime.combine(date, datetime.time(hour, minute))
                    if run_time > now:
                        return run_time
        return None

    def describe(self) -> str | None:
        """Describe a daily or weekly schedule in a human-readable format.

        :return: the description, e.g. ``every day at 02:00``, or ``None`` for other kinds of schedules
        """
        if (
            len(self.minutes) != 1
            or len(self.hours) != 1
            or self.days_restricted
            or self.months != frozenset(range(1, 13))
        ):
            return None
        time_str = f"{next(iter(self.hours)):02d}:{next(iter(self.minutes)):02d}"
        if not self.weekdays_restricted:
            return f"every day at {time_str}"
        weekdays = ", ".join(WEEKDAY_NAMES[weekday] for weekday in sorted(self.weekdays))
        return f"on {weekdays} at {time_str}"


# -- Utilities ---------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Format a duration in a human-readable format.

    :param seconds: the duration in seconds
    :return: the duration, e.g. ``1 d 2 h 3 min``
    """
    minutes = int(seconds) // 60
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    return f"{days} d {hours} h {minutes} min" if days else f"{hours} h {minutes} min"


def format_time(value: datetime.datetime) -> str:
    """Format a local time in a human-readable format.

    :param value: the local time without time zone information
    :return: the time, e.g. ``Sun 2026-10-04 12:25:06 EEST``
    """
    return value.astimezone().strftime("%a %Y-%m-%d %H:%M:%S %Z")


def log_files(repo: Repo) -> list[Path]:
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


def tail(path: Path, lines: int) -> list[str]:
    """Read the last lines of a file without reading the whole file.

    :param path: the path of the file
    :param lines: the number of lines
    :return: the last lines
    """
    block_size = 8192
    with path.open("rb") as file:
        position = file.seek(0, os.SEEK_END)
        data = b""
        while position > 0 and data.count(b"\n") <= lines:
            read_size = min(block_size, position)
            position -= read_size
            file.seek(position)
            data = file.read(read_size) + data
    return data.decode(errors="replace").splitlines()[-lines:]


def read_lock(lock_file: Path) -> tuple[bool, dict[str, Any] | None]:
    """Check whether a lock is held, and read the information of the current or latest run from it.

    The lock is checked by trying to take a shared lock without blocking.

    :param lock_file: the lock file, e.g. :attr:`Repo.lock_file` or :data:`SEQUENCE_LOCK_FILE`
    :return: whether the lock is held, and the information on the current or latest run, if available
    """
    try:
        file = lock_file.open()
    except FileNotFoundError:
        return False, None
    with file:
        try:
            fcntl.flock(file, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            locked = True
        else:
            locked = False
            fcntl.flock(file, fcntl.LOCK_UN)
        try:
            info = json.loads(file.read())
        except json.JSONDecodeError:
            info = None
    return locked, info if isinstance(info, dict) else None


def read_lock_file(repo: Repo) -> tuple[bool, LockInfo | None]:
    """Check whether the lock of the nightly run of a repository is held, and read the information from it.

    :param repo: the repository
    :return: whether the lock is held, and the information on the current or latest run, if available
    """
    locked, info = read_lock(repo.lock_file)
    return locked, cast(LockInfo | None, info)


# -- Processes ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ProcessInfo:
    """Information on a process from ``/proc/PID/stat``, see proc_pid_stat(5)."""

    pid: int
    ppid: int
    #: The state, e.g. ``R`` for running, ``S`` for sleeping and ``Z`` for zombie
    state: str
    #: The start time in clock ticks after the system boot.
    #: Together with the PID, this identifies the process even if the PID is reused.
    start_time: int


def read_process(pid: int) -> ProcessInfo | None:
    """Read the information on a process.

    :param pid: the process ID
    :return: the information, or ``None`` if the process does not exist
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name is in parentheses and may contain spaces and parentheses itself,
    # so the fields are counted from the last closing parenthesis.
    # The fields after it are numbered from 3 in proc_pid_stat(5).
    fields = stat[stat.rindex(")") + 2 :].split()
    return ProcessInfo(pid=pid, ppid=int(fields[1]), state=fields[0], start_time=int(fields[19]))


def read_cmdline(pid: int) -> list[str]:
    """Read the command-line arguments of a process.

    :param pid: the process ID
    :return: the arguments, or an empty list if the process does not exist
    """
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    return [arg.decode(errors="replace") for arg in cmdline.split(b"\0") if arg]


def is_alive(process: ProcessInfo) -> bool:
    """Check whether a process is still running.

    :param process: the process
    :return: ``False`` if the process has exited, including if it is a zombie, or if its PID has been reused
    """
    current = read_process(process.pid)
    return current is not None and current.start_time == process.start_time and current.state != "Z"


def find_descendants(roots: collections.abc.Iterable[int]) -> dict[int, ProcessInfo]:
    """Find the child processes of the given processes recursively.

    :param roots: the process IDs of the root processes
    :return: the descendant processes, excluding the root processes
    """
    children: dict[int, list[ProcessInfo]] = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit() and (process := read_process(int(path.name))) is not None:
            children.setdefault(process.ppid, []).append(process)
    descendants: dict[int, ProcessInfo] = {}
    stack = list(roots)
    while stack:
        for child in children.get(stack.pop(), []):
            if child.pid not in descendants:
                descendants[child.pid] = child
                stack.append(child.pid)
    return descendants


def find_run_process(pid: int | None) -> ProcessInfo | None:
    """Find the process of the nightly run that holds a lock.

    :param pid: the process ID read from the lock file
    :return: the process, or ``None`` if the process is not running a nightly run
    """
    if pid is None:
        return None
    cmdline = read_cmdline(pid)
    if "--run" not in cmdline or not any(arg.endswith("nightly.py") for arg in cmdline):
        return None
    return read_process(pid)


def format_process(process: ProcessInfo, max_length: int = 150) -> str:
    """Format a process as its PID and command line.

    :param process: the process
    :param max_length: the maximum length of the command line, after which it is truncated
    :return: the formatted process
    """
    cmdline = " ".join(read_cmdline(process.pid))
    if len(cmdline) > max_length:
        cmdline = cmdline[: max_length - 3] + "..."
    return f"{process.pid:>8} {cmdline}"


# -- Status ------------------------------------------------------------------


def read_crontab_entries() -> list[str] | None:
    """Read the crontab entries of the nightly run.

    :return: the crontab lines that contain :data:`CRONTAB_PATTERN`, or ``None`` if crontab was not found
    """
    try:
        crontab = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return None
    return [line for line in crontab.splitlines() if CRONTAB_PATTERN in line]


def next_scheduled_run(now: datetime.datetime) -> datetime.datetime | None:
    """Get the time of the next scheduled nightly run from the crontab.

    :param now: the current local time
    :return: the time of the next run, or ``None`` if no run is scheduled or the schedule could not be parsed
    """
    next_runs = []
    for entry in read_crontab_entries() or []:
        if entry.lstrip().startswith("#") or "--run" not in entry.split():
            continue
        try:
            next_run = CronSchedule.parse(entry).next_run(now)
        except ValueError:
            continue
        if next_run is not None:
            next_runs.append(next_run)
    return min(next_runs, default=None)


def read_skip_request(repo: Repo) -> datetime.datetime | None:
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
    print(f'Crontab entry (crontab -l | grep "{CRONTAB_PATTERN}"):')
    entries = read_crontab_entries()
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
                f"in {format_duration((next_run - now).total_seconds())}"
            )


def status(repos: collections.abc.Sequence[Repo]) -> None:
    """Print the status of the nightly runs.

    :param repos: the repositories
    """
    now = datetime.datetime.now()
    print(f"The nightly run is run with: {RUN_COMMAND}")
    print(f"Current time: {format_time(now)}")
    print()
    print_crontab()
    print(f"Repositories, in the order of the runs: {', '.join(repo.name for repo in repos)}")
    for repo in repos:
        print()
        print(f"=== {repo.name} ({repo.path}) ===")
        repo_status(repo, now)


def repo_status(repo: Repo, now: datetime.datetime) -> None:
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
            print(f"  Running for: {format_duration((now - started).total_seconds())}")
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
        print(f"  The run can be stopped with: {SCRIPT_PATH} --stop")

    logs = log_files(repo)
    if not logs:
        print(f"Last lines of the latest log: (no logs in {repo.log_dir})")
        return
    print(f"Last lines of the latest log {logs[-1]}:")
    for line in tail(logs[-1], 5):
        print(f"  {line}")


# -- Nightly run -------------------------------------------------------------


def create_log_file(repo: Repo, date: datetime.date) -> Path:
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


def latest_log_file(repo: Repo, date: datetime.date) -> Path:
    """The latest log file of a repository of a date, to which messages of the runs that were not started are appended.

    :param repo: the repository
    :param date: the date
    :return: the path of the latest existing log file of the date, or of a new log file if there is none
    """
    logs = [log_file for log_file in log_files(repo) if log_file.name.startswith(f"nightly_{date:%Y-%m-%d}")]
    if logs:
        return logs[-1]
    return create_log_file(repo, date)


def log(message: str) -> None:
    """Print a message with a timestamp.

    :param message: the message
    """
    print(f"[{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def append_log(log_file: Path, message: str) -> None:
    """Append a message with a timestamp to a log file, without redirecting the output to it.

    :param log_file: the path of the log file
    :param message: the message
    """
    with log_file.open("a") as file:
        print(f"[{datetime.datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", file=file)


def redirect_output(log_file: Path) -> None:
    """Redirect stdout and stderr of this process and of all its subprocesses to the log file.

    :param log_file: the path of the log file, to which the output is appended
    """
    sys.stdout.flush()
    sys.stderr.flush()
    fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o666)
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())
    os.close(fd)


def acquire_lock(lock_file: Path) -> int | None:
    """Take a lock without blocking.

    The lock is released when the returned file descriptor is closed,
    which happens at the latest when this process exits.
    The file descriptor is not inherited by the subprocesses.

    :param lock_file: the lock file, e.g. :attr:`Repo.lock_file` or :data:`SEQUENCE_LOCK_FILE`
    :return: the file descriptor of the lock file, or ``None`` if another run holds the lock
    """
    fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def write_lock_info(fd: int, info: collections.abc.Mapping[str, object]) -> None:
    """Write the information on the current run to a lock file.

    :param fd: the file descriptor of the lock file
    :param info: the information on the current run
    """
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, (json.dumps(info, indent=2) + "\n").encode())


def git(repo: Repo, *args: str) -> str:
    """Run a Git command in the repository.

    :param repo: the repository
    :param args: the arguments of the Git command
    :return: the stripped standard output of the command
    """
    return subprocess.run(["git", *args], cwd=repo.path, stdout=subprocess.PIPE, text=True, check=True).stdout.strip()


def find_uv() -> str | None:
    """Find uv, which may not be in the PATH of cron.

    Cron runs with a minimal PATH, which does not include the user-specific directories
    where the standalone installer of uv puts it.
    ``$UV_INSTALL_DIR`` and ``$XDG_BIN_HOME`` are the custom installation directories supported by the installer,
    ``~/.local/bin`` is its default, and ``~/.cargo/bin`` is the default of older uv versions.
    The existing directories are added to the PATH of this process and its subprocesses.
    Therefore, this should be called only once.

    :return: the path of uv, or ``None`` if it was not found
    """
    home = Path.home()
    dirs = [home / ".cargo" / "bin", home / ".local" / "bin"]
    dirs += [Path(os.environ[name]) for name in ("XDG_BIN_HOME", "UV_INSTALL_DIR") if os.environ.get(name)]
    path = os.environ.get("PATH", "")
    for directory in dirs:
        if directory.is_dir():
            path = f"{directory}{os.pathsep}{path}"
    os.environ["PATH"] = path
    return shutil.which("uv")


def archive(archive_path: Path, paths: dict[Path, str], exclude: collections.abc.Iterable[str] = ()) -> Path:
    """Compress directories and files to a 7-Zip archive.

    7-Zip appends to an existing archive instead of replacing it,
    so if there is already an archive with the same name, a suffix _2, _3 etc. is added to the filename.
    A message is logged after the archive has been created.

    :param archive_path: the path of the archive
    :param paths: the directories and files to compress, and their names at the root of the archive
    :param exclude: the wildcards of the filenames to leave out of the archive, e.g. ``*.h5``
    :return: the path of the created archive
    :raises FileNotFoundError: if a directory or a file does not exist
    """
    for item in paths:
        if not item.exists():
            raise FileNotFoundError(f"{item} does not exist.")
    path = archive_path
    i = 2
    while path.exists():
        path = archive_path.with_name(f"{archive_path.stem}_{i}{archive_path.suffix}")
        i += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    exclude_args = [f"-xr!{pattern}" for pattern in exclude]
    subprocess.run(["7z", "a", "-mx=9", *exclude_args, str(path), *map(str, paths)], check=True)
    # 7-Zip stores the directories and files with their own names, so they are renamed afterwards.
    # Renaming a directory renames also its contents.
    renames = [arg for item, name in paths.items() if item.name != name for arg in (item.name, name)]
    if renames:
        subprocess.run(["7z", "rn", str(path), *renames], check=True)
    log(f"Created the archive {path} of {', '.join(f'{item} as {name}' for item, name in paths.items())}.")
    return path


def nightly_run(repo: Repo, uv: str | None, lock_fd: int, log_file: Path, started: datetime.datetime) -> int:
    """Run the unit tests, build the documentation and the wheel, and archive the results of a repository.

    The completion of the run is logged before archiving, so that the archived log includes it.
    The creation of the archive is then logged only to the log file.

    :param repo: the repository
    :param uv: the path of uv from :func:`find_uv`
    :param lock_fd: the file descriptor of the lock file of the repository
    :param log_file: the path of the log file
    :param started: the start time of the run
    :return: the exit code
    """
    head = git(repo, "rev-parse", "--short", "HEAD")
    write_lock_info(
        lock_fd, {"pid": os.getpid(), "started": started.isoformat(), "head": head, "log_file": str(log_file)}
    )

    skip_requested = read_skip_request(repo)
    if skip_requested is not None:
        repo.skip_file.unlink(missing_ok=True)
        log(f"Skipping the nightly run, as requested with --skip at {skip_requested.isoformat(timespec='seconds')}.")
        return 0

    # Run only if the HEAD commit is recent, so that unchanged code is not rebuilt every night.
    max_commit_age = int(os.environ.get("MAX_COMMIT_AGE", DEFAULT_MAX_COMMIT_AGE))
    commit_age = time.time() - int(git(repo, "log", "-1", "--format=%ct", "HEAD"))
    if commit_age > max_commit_age:
        log(f"HEAD ({head}) is {int(commit_age // 3600)} h old. Skipping the nightly run.")
        return 0

    log(f"Starting the nightly run of {repo.path} at HEAD {head}.")

    # The build directory is cleaned also by the make target,
    # but cleaning it already here ensures that no output of a previous run is left there if the run fails.
    for directory in (repo.docs_build_dir, repo.dist_dir):
        if directory.exists():
            shutil.rmtree(directory)
            log(f"Cleaned {directory}.")
    fig_dir = repo.fig_parent_dir / f"fig_{started:%Y-%m-%d}"
    # The examples save their figures in this directory.
    env = {**os.environ, repo.fig_dir_env: str(fig_dir)}
    log(f"Saving the figures to {fig_dir} (${repo.fig_dir_env}).")

    if uv is None:
        log(f"uv was not found in PATH: {os.environ['PATH']}")
        return 1
    uv_version = subprocess.run([uv, "--version"], stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    log(f"Using {uv} ({uv_version}).")

    # Run in the virtualenv that uv manages.
    # The --frozen ensures that the locked dependency versions are used as they are, without updating uv.lock.
    uv_run = [uv, "run", "--frozen", "--project", str(repo.path)]

    subprocess.run([*uv_run, "pytest"], cwd=repo.path, env=env, check=True)
    log("Unit tests finished.")

    # A variable given on the command line of make overrides the one in the Makefile,
    # and it is passed on to the recursive make calls.
    subprocess.run(
        [*uv_run, "make", "-C", str(repo.docs_dir), DOCS_TARGET, f"BUILDDIR={repo.docs_build_dir}"],
        cwd=repo.path,
        env=env,
        check=True,
    )
    log("Documentation build finished.")

    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(repo.dist_dir), str(repo.path)], cwd=repo.path, env=env, check=True
    )
    log(f"Wheel build finished: {', '.join(wheel.name for wheel in sorted(repo.dist_dir.glob('*.whl')))}.")

    # This is logged before archiving, so that the archived log includes it.
    log("Nightly run finished. Archiving the results.")
    # A copy of the log is archived, as the output of 7-Zip is written to the log while it's being archived.
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_copy = Path(tmp_dir) / log_file.name
        shutil.copy2(log_file, log_copy)
        archive(
            repo.archive_dir / f"nightly_{started:%Y-%m-%d}.7z",
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


def run_repo(repo: Repo, uv: str | None, original_stdout: TextIO, original_stderr: TextIO) -> int:
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
        log(f"FAILED: {' '.join(map(str, error.cmd))} returned exit code {error.returncode}.")
        exit_code = error.returncode
    except StopRequested:
        log("Stopped by SIGTERM, e.g. by --stop.")
        stopped = True
        raise
    except Exception:  # noqa: BLE001
        # The runs of the other repositories are continued after an unexpected error.
        log("FAILED with the following exception.")
        traceback.print_exc()
        sys.stderr.flush()
    except BaseException:
        log("FAILED with the following exception.")
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


def run(repos: collections.abc.Sequence[Repo]) -> int:
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


def send_signal(processes: collections.abc.Iterable[ProcessInfo], sig: signal.Signals) -> int:
    """Send a signal to those of the processes that are still running.

    :param processes: the processes
    :param sig: the signal
    :return: the number of processes to which the signal was sent
    """
    count = 0
    for process in processes:
        if is_alive(process):
            try:
                os.kill(process.pid, sig)
            except ProcessLookupError:
                continue
            count += 1
    return count


def wait_for_exit(targets: dict[int, ProcessInfo], timeout: float, sig: signal.Signals) -> list[ProcessInfo]:
    """Wait for the processes to exit.

    The child processes that are created in the meantime are added to the targets, and the signal is sent to them.

    :param targets: the processes, which are updated in place
    :param timeout: the maximum waiting time in seconds
    :param sig: the signal to send to the new child processes
    :return: the processes that are still running
    """
    deadline = time.monotonic() + timeout
    while True:
        alive = [process for process in targets.values() if is_alive(process)]
        new = {
            pid: process
            for pid, process in find_descendants(process.pid for process in alive).items()
            if pid not in targets
        }
        targets.update(new)
        send_signal(new.values(), sig)
        alive += new.values()
        if not alive or time.monotonic() > deadline:
            return alive
        time.sleep(0.1)


def stop(repos: collections.abc.Sequence[Repo], timeout: float) -> int:
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
    # The nightly run process is signalled first, so that it does not start new steps
    # when its child processes exit.
    sent = send_signal(targets.values(), signal.SIGTERM)
    print(f"Sent SIGTERM to {sent} processes. Waiting up to {timeout} s for them to exit.")
    alive = wait_for_exit(targets, timeout, signal.SIGTERM)
    if alive:
        print(f"Sending SIGKILL to {len(alive)} processes that did not exit:")
        for target in alive:
            print(format_process(target))
        send_signal(alive, signal.SIGKILL)
        alive = wait_for_exit(targets, 5, signal.SIGKILL)
    if alive:
        print(f"Failed to stop {len(alive)} processes:")
        for target in alive:
            print(format_process(target))
        return 1

    for log_file in log_files:
        if log_file.exists():
            append_log(
                log_file, f"The nightly run and its {len(targets) - 1} child processes were stopped with --stop."
            )
    print(f"The nightly run and its {len(targets) - 1} child processes were stopped.")
    return 0


# -- Skipping ----------------------------------------------------------------


def skip(repos: collections.abc.Sequence[Repo]) -> int:
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
        print(f"Next scheduled run: {format_time(next_run)}, in {format_duration((next_run - now).total_seconds())}")
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
    repos = [Repo(path.resolve()) for path in args.repos] if args.repos else default_repos()
    for repo in repos:
        if not repo.path.is_dir():
            parser.error(f"The repository {repo.path} does not exist.")
    if args.run:
        return run(repos)
    if args.stop:
        return stop(repos, args.timeout)
    if args.skip:
        return skip(repos)
    status(repos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
