"""Tests for the time utilities."""

import datetime
import importlib.util
import sys
import unittest

import pytest

from pttools.utils.time import format_duration, format_time, log_duration


@log_duration
def add(a: int, b: int = 1) -> int:
    """Add two numbers."""
    return a + b


@log_duration
def fail() -> None:
    """Raise an error."""
    raise ValueError("Failed")


class FormatDurationTest(unittest.TestCase):
    """Tests for formatting durations."""

    def test_format(self) -> None:
        """The leading zero units should be omitted, and the seconds should have three decimals."""
        cases = (
            (0, "0.000 s"),
            (3.1234, "3.123 s"),
            (0.0005, "0.001 s"),
            (59.9996, "1 min 0.000 s"),
            (123.123, "2 min 3.123 s"),
            (3723.123, "1 h 2 min 3.123 s"),
            (3603, "1 h 0 min 3.000 s"),
            (90000, "1 d 1 h 0 min 0.000 s"),
            (-123.123, "-2 min 3.123 s"),
            (-1e-4, "0.000 s"),
            (float("inf"), "inf s"),
            (float("nan"), "nan s"),
        )
        for seconds, expected in cases:
            with self.subTest(seconds=seconds):
                assert format_duration(seconds) == expected

    @staticmethod
    def test_timedelta() -> None:
        """A timedelta should be formatted the same way as the corresponding number of seconds."""
        assert format_duration(datetime.timedelta(hours=1, minutes=2, seconds=3, milliseconds=123)) \
            == "1 h 2 min 3.123 s"
        assert format_duration(datetime.timedelta(days=1)) == "1 d 0 h 0 min 0.000 s"


    def test_days_and_minutes(self) -> None:
        """The days should be included in the hours with days=False, and the seconds omitted with seconds=False."""
        cases = (
            (0, {"seconds": False}, "0 min"),
            (59.9, {"seconds": False}, "0 min"),
            (3 * 3600 + 61, {"seconds": False}, "3 h 1 min"),
            (90000, {"days": False}, "25 h 0 min 0.000 s"),
            (2 * 86400 + 3600, {"seconds": False}, "2 d 1 h 0 min"),
            (86400 + 59, {"seconds": False}, "1 d 0 h 0 min"),
            (-90, {"seconds": False}, "-1 min"),
            (-30, {"seconds": False}, "0 min"),
            (float("inf"), {"seconds": False}, "inf min"),
        )
        for seconds, kwargs, expected in cases:
            with self.subTest(duration=seconds, **kwargs):
                assert format_duration(seconds, **kwargs) == expected


def test_format_time() -> None:
    """A time should be formatted with the day of the week and the time zone."""
    value = datetime.datetime(2026, 10, 4, 12, 25, 6)
    assert format_time(value) == f"Sun 2026-10-04 12:25:06 {value.astimezone().strftime('%Z')}"
    utc = datetime.datetime(2026, 10, 4, 12, 25, 6, tzinfo=datetime.UTC)
    assert format_time(utc) == utc.astimezone().strftime("%a %Y-%m-%d %H:%M:%S %Z")


class LogDurationTest(unittest.TestCase):
    """Tests for logging the time spent by a function."""

    def test_log(self) -> None:
        """The name and the file of the function and the time should be logged, and the result returned."""
        with self.assertLogs(__name__, level="INFO") as logs:
            assert add(1, b=2) == 3
        assert len(logs.records) == 1
        message = logs.records[0].getMessage()
        assert message.startswith("add in test_time.py took ")
        assert message.endswith(" s.")

    def test_exception(self) -> None:
        """The time should be logged also when the function raises an exception, which should be propagated."""
        with self.assertLogs(__name__, level="INFO") as logs, pytest.raises(ValueError, match="Failed"):
            fail()
        assert logs.records[0].getMessage().startswith("fail in test_time.py failed after ")

    def test_sphinx_gallery(self) -> None:
        """The file name should be found also in the way Sphinx-Gallery runs the examples.

        Sphinx-Gallery compiles the code with the path of the example as the file name,
        and runs it in a new ``__main__`` module that does not define ``__file__``.
        """
        code = (
            "from pttools.utils.time import log_duration\n"
            "@log_duration\n"
            "def main():\n"
            "    return 1\n"
            "result = main()\n"
        )
        spec = importlib.util.spec_from_loader("__main__", None)
        assert spec is not None
        fake_main = importlib.util.module_from_spec(spec)
        orig_main = sys.modules["__main__"]
        sys.modules["__main__"] = fake_main
        try:
            with self.assertLogs("__main__", level="INFO") as logs:
                exec(compile(code, "/path/to/example.py", "exec"), fake_main.__dict__)
        finally:
            sys.modules["__main__"] = orig_main
        assert "__file__" not in fake_main.__dict__
        assert fake_main.__dict__["result"] == 1
        assert logs.records[0].getMessage().startswith("main in example.py took ")

    @staticmethod
    def test_wraps() -> None:
        """The decorated function should keep the name and the docstring of the original function."""
        assert add.__name__ == "add"
        assert add.__doc__ == "Add two numbers."
