"""Tests for the exception utilities."""

import logging

import pytest

from pttools.utils.exceptions import log_exceptions

logger: logging.Logger = logging.getLogger(__name__)


def test_log_exceptions(caplog: pytest.LogCaptureFixture) -> None:
    """The exceptions should be logged with their tracebacks instead of being raised."""
    with caplog.at_level(logging.ERROR, logger=__name__), log_exceptions(logger, "Failed to do %s", "something"):
        raise ValueError("Test error")
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.getMessage() == "Failed to do something"
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], ValueError)


def test_log_exceptions_no_exception(caplog: pytest.LogCaptureFixture) -> None:
    """Nothing should be logged if there is no exception."""
    with caplog.at_level(logging.ERROR, logger=__name__), log_exceptions(logger, "Failed"):
        pass
    assert not caplog.records


def test_log_exceptions_keyboard_interrupt() -> None:
    """Exceptions that are not subclasses of Exception should be raised, so that the script can be stopped."""
    with pytest.raises(KeyboardInterrupt), log_exceptions(logger, "Failed"):
        raise KeyboardInterrupt
