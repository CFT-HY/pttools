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
import subprocess
import sys
import tarfile
import time
from typing import Any
import zlib

import pytest

pytest.importorskip("fcntl", reason="The nightly run script uses fcntl, which is not available on Windows.")

import fcntl

from pttools.docs import nightly
from pttools.docs.nightly import CronSchedule, Repo

requires_optipng: pytest.MarkDecorator = pytest.mark.skipif(
    shutil.which("optipng") is None, reason="optipng is not installed"
)
requires_xz: pytest.MarkDecorator = pytest.mark.skipif(shutil.which("xz") is None, reason="xz is not installed")

# A Sunday
NOW: datetime.datetime = datetime.datetime(2026, 10, 4, 12, 25, 6)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("*", set(range(24))),
        ("5", {5}),
        ("1-3", {1, 2, 3}),
        ("*/6", {0, 6, 12, 18}),
        ("1-10/4", {1, 5, 9}),
        ("20/2", {20, 22}),
        ("1,3,5-6", {1, 3, 5, 6}),
    ],
)
def test_parse_cron_field(field: str, expected: set[int]) -> None:
    """The supported cron field syntaxes should be parsed to the matching values."""
    assert nightly.parse_cron_field(field, 0, 23) == expected


@pytest.mark.parametrize("field", ["24", "5-3", "*/0", "mon", ""])
def test_parse_cron_field_invalid(field: str) -> None:
    """Invalid and unsupported fields should raise a ValueError."""
    with pytest.raises(ValueError):  # noqa: PT011
        nightly.parse_cron_field(field, 0, 23)


@pytest.mark.parametrize(
    ("entry", "description", "next_run"),
    [
        ("0 2 * * * /path/pttools/docs/nightly.py --run", "every day at 02:00", datetime.datetime(2026, 10, 5, 2, 0)),
        ("30 13 * * * cmd", "every day at 13:30", datetime.datetime(2026, 10, 4, 13, 30)),
        ("0 2 * * 1-5 cmd", "on Mon, Tue, Wed, Thu, Fri at 02:00", datetime.datetime(2026, 10, 5, 2, 0)),
        ("0 2 * * 6,7 cmd", "on Sun, Sat at 02:00", datetime.datetime(2026, 10, 10, 2, 0)),
        ("@daily cmd", "every day at 00:00", datetime.datetime(2026, 10, 5, 0, 0)),
        ("*/15 * * * * cmd", None, datetime.datetime(2026, 10, 4, 12, 30)),
        ("0 3 1 * * cmd", None, datetime.datetime(2026, 11, 1, 3, 0)),
        # When both the day of the month and the day of the week are restricted, either of them has to match.
        ("0 3 10 * 1 cmd", None, datetime.datetime(2026, 10, 5, 3, 0)),
        ("0 0 29 2 * cmd", None, datetime.datetime(2028, 2, 29, 0, 0)),
    ],
)
def test_cron_schedule(entry: str, description: str | None, next_run: datetime.datetime) -> None:
    """The description and the next run time should match those of cron."""
    schedule = CronSchedule.parse(entry)
    assert schedule.describe() == description
    assert schedule.next_run(NOW) == next_run


@pytest.mark.parametrize("entry", ["0 2 * *", "@reboot cmd", "0 2 * jan * cmd"])
def test_cron_schedule_invalid(entry: str) -> None:
    """Invalid and unsupported schedules should raise a ValueError."""
    with pytest.raises(ValueError):  # noqa: PT011
        CronSchedule.parse(entry)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0 h 0 min"), (59, "0 h 0 min"), (3 * 3600 + 61, "3 h 1 min"), (2 * 86400 + 3600, "2 d 1 h 0 min")],
)
def test_format_duration(seconds: float, expected: str) -> None:
    """Durations should be formatted with days only when needed."""
    assert nightly.format_duration(seconds) == expected


@pytest.mark.parametrize("n_lines", [0, 3, 10_000])
def test_tail(tmp_path: Path, n_lines: int) -> None:
    """The last lines should be returned also when they span multiple blocks."""
    path = tmp_path / "test.log"
    lines = [f"line {i} " + "x" * 100 for i in range(n_lines)]
    path.write_text("".join(f"{line}\n" for line in lines))
    assert nightly.tail(path, 5) == lines[-5:]


def test_create_log_file(tmp_path: Path) -> None:
    """A new log file should be created for each run, with a suffix for the later runs of the same day."""
    repo = Repo(tmp_path)
    names = [nightly.create_log_file(repo, NOW.date()).name for _ in range(3)]
    assert names == ["nightly_2026-10-04.log", "nightly_2026-10-04_2.log", "nightly_2026-10-04_3.log"]
    assert nightly.create_log_file(repo, datetime.date(2026, 10, 5)).name == "nightly_2026-10-05.log"


def test_log_files(tmp_path: Path) -> None:
    """The log files should be in the order of the runs, also when the suffix has multiple digits."""
    repo = Repo(tmp_path)
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
    repo = Repo(tmp_path)
    assert nightly.read_lock_file(repo) == (False, None)

    info: nightly.LockInfo = {"pid": 123, "started": NOW.isoformat(), "head": "abc1234", "log_file": "test.log"}
    fd = nightly.acquire_lock(repo.lock_file)
    assert fd is not None
    try:
        nightly.write_lock_info(fd, info)
        assert nightly.read_lock_file(repo) == (True, info)
        # A second run should not get the lock.
        assert nightly.acquire_lock(repo.lock_file) is None
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
def test_archive(tmp_path: Path) -> None:
    """The archive should contain the given paths with their new names, without the excluded files.

    An existing archive should not be overwritten.
    """
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "a.txt").write_text("a")
    (src / "sub" / "data.h5").write_text("h5")
    (src / "sub" / "data.h5.sha256").write_text("sha")
    log_file = tmp_path / "run.log"
    log_file.write_text("log")
    paths = {src: "renamed", log_file: "run.log"}
    archive_path = tmp_path / "out" / "repo_nightly_2026-10-04.tar.xz"

    first = nightly.archive(archive_path, paths, exclude=("*.h5",))
    assert first == archive_path
    with tarfile.open(first) as tar:
        assert set(tar.getnames()) == {
            "renamed", "renamed/a.txt", "renamed/sub", "renamed/sub/data.h5.sha256", "run.log"
        }
    second = nightly.archive(archive_path, paths)
    assert second.name == "repo_nightly_2026-10-04_2.tar.xz"
    with tarfile.open(second) as tar:
        assert "renamed/sub/data.h5" in tar.getnames()
    assert nightly.archive(archive_path, paths).name == "repo_nightly_2026-10-04_3.tar.xz"

    with pytest.raises(FileNotFoundError):
        nightly.archive(archive_path, {tmp_path / "missing": "missing"})


def test_archive_xz_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If xz fails, the error should be raised and the partial archive removed."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_xz = fake_bin / "xz"
    fake_xz.write_text("#!/bin/sh\ncat > /dev/null\nexit 2\n")
    fake_xz.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    src = tmp_path / "a.txt"
    src.write_text("a")
    archive_path = tmp_path / "repo_nightly_2026-10-04.tar.xz"
    with pytest.raises(subprocess.CalledProcessError):
        nightly.archive(archive_path, {src: "a.txt"})
    assert not archive_path.exists()



def test_physical_cpu_count() -> None:
    """The number of physical cores should be at least 1 and at most the number of logical CPUs."""
    assert 1 <= nightly.physical_cpu_count() <= (os.cpu_count() or 1)


@pytest.fixture
def no_sys_topology(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the CPU topology in /sys unavailable."""
    read_text = Path.read_text

    def fake_read_text(self: Path, *args: Any, **kwargs: Any) -> str:
        if str(self).startswith("/sys/devices/system/cpu/"):
            raise FileNotFoundError(self)
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)


@pytest.mark.usefixtures("no_sys_topology")
def test_physical_cpu_count_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the CPU topology in /sys, the physical cores should be counted with psutil."""
    psutil = pytest.importorskip("psutil")
    monkeypatch.setattr(psutil, "cpu_count", lambda logical=True: 3 if not logical else 6)
    assert nightly.physical_cpu_count() == 3


@pytest.mark.usefixtures("no_sys_topology")
def test_physical_cpu_count_without_psutil(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without the CPU topology in /sys and psutil, the number of logical CPUs should be returned.

    An error with installation instructions should be logged.
    """
    # None in sys.modules makes the import raise an ImportError.
    monkeypatch.setitem(sys.modules, "psutil", None)
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    assert nightly.physical_cpu_count() == cpus
    output = capsys.readouterr().out
    assert "ERROR: The CPU topology is not available in /sys, and psutil is not installed" in output
    assert "sudo apt install python3-psutil" in output


def create_png(path: Path, width: int = 64, height: int = 64) -> None:
    """Create an uncompressed grayscale PNG file, which optipng can make smaller.

    :param path: the path of the file
    :param width: the width of the image in pixels
    :param height: the height of the image in pixels
    """
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            len(data).to_bytes(4, "big") + chunk_type + data
            + zlib.crc32(chunk_type + data).to_bytes(4, "big")
        )

    pixels = b"".join(b"\x00" + bytes((x * y) % 256 for x in range(width)) for y in range(height))
    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes((8, 0, 0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(pixels, 0)) + chunk(b"IEND", b"")
    )


@requires_optipng
def test_optimize_pngs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The PNG files should be made smaller, and identical files should stay identical.

    Invalid files should be logged and left as they are.
    """
    build_dir = tmp_path / "_build"
    fig_dir = tmp_path / "fig"
    create_png(build_dir / "html" / "a.png")
    create_png(build_dir / "epub" / "a.png")
    create_png(fig_dir / "b.png", width=32)
    invalid = fig_dir / "invalid.png"
    invalid.write_text("not a PNG")
    (fig_dir / "notes.txt").write_text("text")
    sizes_before = {path: path.stat().st_size for path in tmp_path.rglob("*.png")}

    nightly.optimize_pngs([build_dir, fig_dir, tmp_path / "missing"])

    output = capsys.readouterr().out
    assert f"optipng failed for {invalid}" in output
    assert "Optimized 4 PNG files (3 unique) with optipng -o7 -preserve -quiet" in output
    assert f"{build_dir} " in output
    assert f"{fig_dir} " in output
    assert invalid.read_text() == "not a PNG"
    for path, size in sizes_before.items():
        if path != invalid:
            assert path.stat().st_size < size, path
    assert (build_dir / "html" / "a.png").read_bytes() == (build_dir / "epub" / "a.png").read_bytes()


def test_optimize_pngs_without_optipng(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    """Without optipng, the PNG files should be left as they are."""
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    png = tmp_path / "fig" / "a.png"
    create_png(png)
    data = png.read_bytes()
    nightly.optimize_pngs([png.parent])
    assert "optipng was not found in PATH" in capsys.readouterr().out
    assert png.read_bytes() == data

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
                descendants = nightly.find_descendants([run.pid])
                if sum(nightly.read_cmdline(pid) == ["sleep", "1000"] for pid in descendants) == 2:
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
            nightly.send_signal([*nightly.find_descendants([run.pid]).values()], nightly.signal.SIGKILL)
            run.kill()

    assert not any(nightly.is_alive(process) for process in descendants.values())
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
    log_file = nightly.create_log_file(Repo(repo), datetime.date.today())
    with (repo / "nightly.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run([sys.executable, script, "--run"], env=env, capture_output=True, text=True, check=False)
    assert (result.returncode, result.stdout, result.stderr) == (1, "", "")
    # The message should be appended to the log of the run in progress instead of a new log.
    assert list((repo / "logs").glob("nightly_*.log")) == [log_file]
    assert "Another nightly run (PID unknown) of repo is in progress. Skipping repo." in log_file.read_text()
