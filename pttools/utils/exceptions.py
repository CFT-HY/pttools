"""Utilities for handling exceptions."""

from __future__ import annotations

from contextlib import contextmanager

from pttools.type_hints import TYPE_CHECKING_OR_DOCS

if TYPE_CHECKING_OR_DOCS:
    from collections.abc import Iterator
    from logging import Logger

__all__ = ["log_exceptions"]


@contextmanager
def log_exceptions(logger: Logger, msg: str, *args: object) -> Iterator[None]:
    """Log the exceptions raised within the context instead of raising them.

    This is for long scripts, e.g. those that create many figures,
    which should continue with the rest of their work if a part of it fails.
    Only the subclasses of :py:class:`Exception` are caught, so e.g. :py:class:`KeyboardInterrupt` is not.

    :param logger: Logger to which the exceptions are logged
    :param msg: Message of the log entry, with ``%`` placeholders for the arguments
    :param args: Arguments of the message
    """
    try:
        yield
    except Exception:
        logger.exception(msg, *args)
