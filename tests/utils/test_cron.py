"""Tests for the cron utilities."""

import datetime
import os
from pathlib import Path

import pytest

from pttools.utils.cron import CronSchedule, parse_cron_field, read_crontab_entries
from tests.utils.mark import requires_sh

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
    assert parse_cron_field(field, 0, 23) == expected


@pytest.mark.parametrize("field", ["24", "5-3", "*/0", "mon", ""])
def test_parse_cron_field_invalid(field: str) -> None:
    """Invalid and unsupported fields should raise a ValueError."""
    with pytest.raises(ValueError):  # noqa: PT011
        parse_cron_field(field, 0, 23)


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


@requires_sh
def test_read_crontab_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The lines of the crontab should be returned, optionally only those that match the pattern."""
    fake_crontab = tmp_path / "crontab"
    fake_crontab.write_text('#!/bin/sh\nprintf "# comment\\n0 2 * * * job --run\\n@daily other\\n"\n')
    fake_crontab.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    assert read_crontab_entries() == ["# comment", "0 2 * * * job --run", "@daily other"]
    assert read_crontab_entries(r"j[o]b") == ["0 2 * * * job --run"]
    monkeypatch.setenv("PATH", str(tmp_path / "missing"))
    assert read_crontab_entries() is None
