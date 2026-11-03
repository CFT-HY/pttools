"""Parsing of cron schedules and the entries of the crontab of the user."""

import dataclasses
import datetime
import re
import subprocess
from typing import Self

__all__ = ["CRON_NICKNAMES", "WEEKDAY_NAMES", "CronSchedule", "parse_cron_field", "read_crontab_entries"]

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
#: The abbreviated names of the days of the week, from Sunday, which is 0 in cron
WEEKDAY_NAMES: tuple[str, ...] = ("Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat")


@dataclasses.dataclass(frozen=True, slots=True)
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


def read_crontab_entries(pattern: str | re.Pattern[str] | None = None) -> list[str] | None:
    """Read the entries of the crontab of the user with ``crontab -l``.

    :param pattern: a regular expression, with which only the matching lines are returned,
        see :func:`re.search`. By default, all the lines are returned.
    :return: the lines, or ``None`` if the crontab command was not found
    """
    try:
        crontab = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False).stdout
    except FileNotFoundError:
        return None
    return [line for line in crontab.splitlines() if pattern is None or re.search(pattern, line)]
