"""Tests for the nightly run script.

The script is for Linux, but most of it works also on macOS.
It uses :mod:`fcntl` for the locks, which is not available on Windows,
and ``/proc`` for finding and stopping the processes, which is available only on Linux.
"""

import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import pytest

pytest.importorskip("fcntl", reason="The nightly run script uses fcntl, which is not available on Windows.")

import fcntl

from pttools.docs import nightly
from pttools.docs.nightly import NightlyRepo
from pttools.utils import process
from pttools.utils.lock import acquire_lock, write_lock_info

requires_xz: pytest.MarkDecorator = pytest.mark.skipif(shutil.which("xz") is None, reason="xz is not installed")

# A Sunday
NOW: datetime.datetime = datetime.datetime(2026, 10, 4, 12, 25, 6)


def test_create_log_file(tmp_path: Path) -> None:
    """A new log file should be created for each run, with a suffix for the later runs of the same day."""
    repo = NightlyRepo(tmp_path)
    names = [nightly.create_log_file(repo, NOW.date()).name for _ in range(3)]
    assert names == ["nightly_2026-10-04.log", "nightly_2026-10-04_2.log", "nightly_2026-10-04_3.log"]
    assert nightly.create_log_file(repo, datetime.date(2026, 10, 5)).name == "nightly_2026-10-05.log"


def test_log_files(tmp_path: Path) -> None:
    """The log files should be in the order of the runs, also when the suffix has multiple digits."""
    repo = NightlyRepo(tmp_path)
    repo.log_dir.mkdir()
    names = [
        "nightly_2026-10-03_2.log",
        "nightly_2026-10-04.log",
        "nightly_2026-10-04_2.log",
        "nightly_2026-10-04_10.log",
        "nightly_2026-10-05.log",
    ]
    for name in [*reversed(names), "nightly_other.log"]:
        (repo.log_dir / name).touch()
    assert [path.name for path in nightly.log_files(repo)] == names
    assert nightly.latest_log_file(repo, NOW.date()).name == "nightly_2026-10-04_10.log"
    # A log file is created if there is none of the date.
    assert nightly.latest_log_file(repo, datetime.date(2026, 10, 6)).name == "nightly_2026-10-06.log"
    assert (repo.log_dir / "nightly_2026-10-06.log").exists()


def test_lock(tmp_path: Path) -> None:
    """The lock should be detected while it is held, and the run information should be readable from the lock file."""
    repo = NightlyRepo(tmp_path)
    assert nightly.read_lock_file(repo) == (False, None)

    info: nightly.LockInfo = {"pid": 123, "started": NOW.isoformat(), "head": "abc1234", "log_file": "test.log"}
    fd = acquire_lock(repo.lock_file)
    assert fd is not None
    try:
        write_lock_info(fd, info)
        assert nightly.read_lock_file(repo) == (True, info)
        # A second run should not get the lock.
        assert acquire_lock(repo.lock_file) is None
    finally:
        os.close(fd)
    assert nightly.read_lock_file(repo) == (False, info)
    assert json.loads((tmp_path / "nightly.lock").read_text()) == info


@pytest.mark.parametrize("argv", [[], ["--status"]])
def test_main_status(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Both --status and no arguments should print the status."""
    calls: list[None] = []
    monkeypatch.setattr(nightly, "status", lambda _repos: calls.append(None))
    assert nightly.main(argv) == 0
    assert len(calls) == 1


@pytest.mark.parametrize(
    "argv", [["--status", "--run"], ["--status", "--stop"], ["--run", "--stop"], ["--run", "--skip"]]
)
def test_main_exclusive(argv: list[str]) -> None:
    """The actions should be mutually exclusive."""
    with pytest.raises(SystemExit):
        nightly.main(argv)


FAKE_UV = """#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "uv (fake)"
  exit 0
fi
# A process that ignores SIGTERM and therefore has to be killed with SIGKILL
sh -c 'trap "" TERM; exec sleep 1000' &
sh -c 'sleep 1000' &
wait
"""


def init_repo(repo: Path) -> Path:
    """Create a Git repository with an empty commit.

    :param repo: the repository directory
    :return: the repository directory
    """
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    git_user = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]
    subprocess.run(["git", *git_user, "commit", "-q", "--allow-empty", "-m", "Test"], cwd=repo, check=True)
    return repo


def create_repo(tmp_path: Path, fake_uv: str) -> tuple[Path, Path, dict[str, str]]:
    """Create a Git repository with a copy of the nightly script and a fake uv.

    :param tmp_path: the temporary directory
    :param fake_uv: the contents of the fake uv script
    :return: the repository directory, the path of the script copy, and the environment variables for the script
    """
    repo = init_repo(tmp_path / "repo")
    (repo / "pttools" / "docs").mkdir(parents=True)
    script = repo / "pttools" / "docs" / "nightly.py"
    script.write_text(nightly.SCRIPT_PATH.read_text())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uv").write_text(fake_uv)
    (fake_bin / "uv").chmod(0o755)
    # The figure directories set by a nightly run that runs these tests are removed,
    # as the fake uv would otherwise use one of them instead of that of the test repository.
    env = {key: value for key, value in os.environ.items() if not key.endswith("_FIG_DIR")}
    # The directory in $UV_INSTALL_DIR is put first in the PATH of the nightly run.
    return repo, script, {**env, "UV_INSTALL_DIR": str(fake_bin)}


#: A fake uv that creates the outputs of the documentation and wheel builds, and succeeds for the other commands.
#: It fails with exit code 3 in the repositories whose name begins with "fail".
FAKE_UV_OUTPUTS = """#!/bin/sh
case "$(basename "$PWD")" in
  fail*) [ "$1" = "--version" ] || exit 3 ;;
esac
# The figure directory is in the variable NAME_FIG_DIR, where NAME is the name of the repository.
fig_dir="$(env | sed -n 's/^[A-Z0-9_]*_FIG_DIR=//p' | head -n 1)"
if [ "$1" = "build" ]; then
  while [ "$#" -gt 0 ]; do
    if [ "$1" = "--out-dir" ]; then
      mkdir -p "$2"
      echo "wheel" > "$2/ptplot_gw-0.0.0-py3-none-any.whl"
    fi
    shift
  done
  exit 0
fi
for arg in "$@"; do
  case "$arg" in
    BUILDDIR=*)
      mkdir -p "${arg#BUILDDIR=}/html"
      echo "docs" > "${arg#BUILDDIR=}/html/index.html"
      mkdir -p "$fig_dir"
      echo "fig" > "$fig_dir/fig.png"
      ;;
  esac
done
"""


@requires_xz
def test_run(tmp_path: Path) -> None:
    """A successful run should archive the documentation, the figures, the wheel and the log.

    The archived log should include the completion of the run but not the creation of the archive,
    which is logged only to the log file.
    """
    repo, script, env = create_repo(tmp_path, FAKE_UV_OUTPUTS)
    result = subprocess.run([sys.executable, script, "--run"], env=env, capture_output=True, text=True, check=False)
    assert (result.returncode, result.stderr) == (0, "")
    archives = list((repo / "docs" / "nightly").glob("repo_nightly_*.tar.xz"))
    assert len(archives) == 1
    log_file = next((repo / "logs").glob("nightly_*.log"))
    with tarfile.open(archives[0]) as tar:
        names = set(tar.getnames())
        archived_log_file = tar.extractfile(log_file.name)
        assert archived_log_file is not None
        archived_log = archived_log_file.read().decode()
    assert {
        "_build/html/index.html", "fig/fig.png", "dist/ptplot_gw-0.0.0-py3-none-any.whl", log_file.name
    } <= names

    assert "Wheel build finished: ptplot_gw-0.0.0-py3-none-any.whl." in archived_log
    assert "Nightly run finished." in archived_log
    assert "Created the archive" not in archived_log
    # The archived log is a copy of the log until the completion of the run.
    assert archived_log.rstrip("\n").endswith("Nightly run finished. Archiving the results.")
    log_text = log_file.read_text()
    assert log_text.startswith(archived_log)
    assert f"Created the archive {archives[0]}" in log_text.removeprefix(archived_log)


def test_skip(tmp_path: Path) -> None:
    """--skip should make the next run skip, and only the next one."""
    # A fake uv that fails, so that a run that is not skipped fails with its exit code.
    repo, script, env = create_repo(tmp_path, '#!/bin/sh\n[ "$1" = "--version" ] && exit 0\nexit 3\n')
    skip_file = repo / "nightly.skip"

    def run_script(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, script, *args], env=env, capture_output=True, text=True, check=False)

    assert "Next run will be skipped: NO" in run_script("--status").stdout
    result = run_script("--skip")
    assert result.returncode == 0
    assert "The next nightly run of repo will be skipped." in result.stdout
    assert skip_file.exists()
    assert "Next run will be skipped: YES" in run_script("--status").stdout
    assert "already been requested" in run_script("--skip").stdout

    # The skipped run should succeed without reporting a failure to cron.
    result = run_script("--run")
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    assert not skip_file.exists()
    log_file = next((repo / "logs").glob("nightly_*.log"))
    assert "as requested with --skip" in log_file.read_text()
    assert "Next run will be skipped: NO" in run_script("--status").stdout

    # The run after the skipped one should not be skipped, and it should have its own log.
    result = run_script("--run")
    assert result.returncode == 3
    assert "The nightly run of repo failed with exit code 3." in result.stderr
    second_log_file = log_file.with_name(f"{log_file.stem}_2.log")
    assert f"See the log {second_log_file}" in result.stderr
    assert "as requested with --skip" not in second_log_file.read_text()
    assert f"Last lines of the latest log {second_log_file}" in run_script("--status").stdout


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Finding the processes requires /proc of Linux.")
def test_stop(tmp_path: Path) -> None:
    """--stop should stop the nightly run and all its child processes, including those that ignore SIGTERM."""
    repo, script, env = create_repo(tmp_path, FAKE_UV)

    with subprocess.Popen(
        [sys.executable, script, "--run"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    ) as run:
        try:
            # Wait for the fake uv to start both sleep processes.
            deadline = time.monotonic() + 30
            while True:
                descendants = process.find_descendants([run.pid])
                if sum(process.read_cmdline(pid) == ["sleep", "1000"] for pid in descendants) == 2:
                    break
                assert time.monotonic() < deadline, "The fake process tree was not created."
                time.sleep(0.1)

            status = subprocess.run([sys.executable, script], capture_output=True, text=True, check=True).stdout
            assert "Nightly run in progress: YES" in status
            assert "sleep 1000" in status

            result = subprocess.run(
                [sys.executable, script, "--stop", "--timeout", "1"],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            assert "Sending SIGKILL to 1 processes" in result.stdout
            assert run.wait(timeout=10) == 128 + 15
            # A deliberately stopped run should not be reported as a failure to cron.
            assert run.stderr is not None
            assert run.stderr.read() == ""
        finally:
            # Clean up if the test failed.
            process.send_signal([*process.find_descendants([run.pid]).values()], signal.SIGKILL)
            run.kill()

    assert not any(process.is_alive(descendant) for descendant in descendants.values())
    log = next((repo / "logs").glob("nightly_*.log")).read_text()
    assert "Stopped by SIGTERM" in log
    assert "were stopped with --stop" in log
    status = subprocess.run([sys.executable, script], capture_output=True, text=True, check=True).stdout
    assert "Nightly run in progress: NO" in status
    stop_again = subprocess.run([sys.executable, script, "--stop"], capture_output=True, text=True, check=True).stdout
    assert "No nightly run is in progress." in stop_again


@requires_xz
def test_run_multiple_repos(tmp_path: Path) -> None:
    """The nightly run should be run for each repository in sequence, even if it fails for a previous repository.

    The outputs of each repository should be saved in that repository,
    and the status should show the last lines of the latest log of each repository.
    """
    _, script, env = create_repo(tmp_path, FAKE_UV_OUTPUTS)
    fail = init_repo(tmp_path / "fail")
    other = init_repo(tmp_path / "other")
    repo_args = ["--repo", str(fail), "--repo", str(other)]
    result = subprocess.run(
        [sys.executable, script, "--run", *repo_args], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 3, result.stderr
    assert "The nightly run of fail failed with exit code 3." in result.stderr
    assert "other" not in result.stderr

    fail_log = next((fail / "logs").glob("nightly_*.log")).read_text()
    assert "FAILED:" in fail_log
    assert not list((fail / "docs" / "nightly").glob("*_nightly_*.tar.xz"))
    other_log = next((other / "logs").glob("nightly_*.log")).read_text()
    assert "Saving the figures to" in other_log
    assert "(OTHER_FIG_DIR)" in other_log.replace("$", "")
    assert len(list((other / "docs" / "nightly").glob("*_nightly_*.tar.xz"))) == 1
    assert len(list((other / "docs" / "nightly" / "fig").glob("fig_*/fig.png"))) == 1

    status = subprocess.run(
        [sys.executable, script, "--status", *repo_args], capture_output=True, text=True, check=True
    ).stdout
    assert status.index("=== fail") < status.index("=== other")
    assert status.count("Last lines of the latest log") == 2
    assert f"Last lines of the latest log {fail / 'logs'}" in status
    assert f"Last lines of the latest log {other / 'logs'}" in status


def test_default_repos(tmp_path: Path) -> None:
    """By default, the nightly run should be run for PTtools and for PTPlot if it exists next to PTtools."""
    _, script, env = create_repo(tmp_path, FAKE_UV)
    status = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True, check=True).stdout
    assert "Repositories, in the order of the runs: repo\n" in status
    assert "=== PTPlot" not in status
    assert status.count("Last lines of the latest log") == 1

    init_repo(tmp_path / "PTPlot")
    status = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True, check=True).stdout
    assert "Repositories, in the order of the runs: repo, PTPlot\n" in status
    assert f"=== PTPlot ({tmp_path / 'PTPlot'}) ===" in status
    assert status.count("Last lines of the latest log") == 2


def test_sequence_lock(tmp_path: Path) -> None:
    """A new sequence of nightly runs should not be started while another one is in progress."""
    repo, script, env = create_repo(tmp_path, FAKE_UV)
    with (repo / "nightly_sequence.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, script, "--run"], env=env, capture_output=True, text=True, check=False)
        assert (result.returncode, result.stdout, result.stderr) == (1, "", "")
        log_files = list((repo / "logs").glob("nightly_*.log"))
        assert len(log_files) == 1
        assert "Another nightly run (PID unknown) is in progress." in log_files[0].read_text()
        # The message of a run that was not started should be appended to the latest log instead of a new one.
        subprocess.run([sys.executable, script, "--run"], env=env, capture_output=True, text=True, check=False)
    assert list((repo / "logs").glob("nightly_*.log")) == log_files
    assert log_files[0].read_text().count("Another nightly run (PID unknown) is in progress.") == 2


def test_repo_lock(tmp_path: Path) -> None:
    """The run of a repository should not be started while another run of it is in progress."""
    repo, script, env = create_repo(tmp_path, FAKE_UV)
    log_file = nightly.create_log_file(NightlyRepo(repo), datetime.date.today())
    with (repo / "nightly.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, script, "--run"], env=env, capture_output=True, text=True, check=False)
    assert (result.returncode, result.stdout, result.stderr) == (1, "", "")
    # The message should be appended to the log of the run in progress instead of a new log.
    assert list((repo / "logs").glob("nightly_*.log")) == [log_file]
    assert "Another nightly run (PID unknown) of repo is in progress. Skipping repo." in log_file.read_text()
