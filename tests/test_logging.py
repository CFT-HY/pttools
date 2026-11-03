"""Tests for the logging utilities."""

import datetime
from pathlib import Path
import re
import subprocess
import sys

import pytest

from pttools.logging import append_log, log_console

#: The format of the timestamped messages
TIMESTAMPED: re.Pattern[str] = re.compile(r"\[\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}\] (.*)")


def test_log_console(capsys: pytest.CaptureFixture[str]) -> None:
    """The message should be printed to stdout with a timestamp of the current local time."""
    before = datetime.datetime.now().astimezone().replace(microsecond=0)
    log_console("Test message")
    output = capsys.readouterr().out
    match = TIMESTAMPED.fullmatch(output.rstrip("\n"))
    assert match is not None, output
    assert match[1] == "Test message"
    assert datetime.datetime.fromisoformat(output[1:26]) >= before


def test_append_log(tmp_path: Path) -> None:
    """The messages should be appended to the log file in the same format as that of log_console."""
    log_file = tmp_path / "test.log"
    log_file.write_text("existing\n")
    append_log(log_file, "first")
    append_log(log_file, "second")
    lines = log_file.read_text().splitlines()
    assert lines[0] == "existing"
    assert [match[1] for line in lines[1:] if (match := TIMESTAMPED.fullmatch(line))] == ["first", "second"]


def test_redirect_output(tmp_path: Path) -> None:
    """The output of the process and of its subprocesses should be appended to the log file."""
    log_file = tmp_path / "test.log"
    log_file.write_text("existing\n")
    code = (
        "import subprocess, sys\n"
        "from pttools.logging import redirect_output\n"
        f"redirect_output({str(log_file)!r})\n"
        "print('stdout', flush=True)\n"
        "print('stderr', file=sys.stderr, flush=True)\n"
        "subprocess.run(['echo', 'child'], check=True)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert (result.stdout, result.stderr) == ("", "")
    assert log_file.read_text().splitlines() == ["existing", "stdout", "stderr", "child"]
