"""Logging configuration."""

import dataclasses
import faulthandler
import logging
import os
from pathlib import Path
from threading import Lock
import time

LOGGING_LOCK: Lock = Lock()

#: Format of the log messages
LOG_FORMAT: str = \
    '%(asctime)s %(levelname)-8s %(module)-20s %(funcName)-32s %(lineno)-4d %(process)-3d %(message)s'


@dataclasses.dataclass(frozen=True)
class LoggingConfig:
    """The logging configuration created by :py:func:`setup_logging`.

    This is picklable so that it can be passed to worker processes
    for replicating the logging configuration of the main process.
    """

    log_file_path: Path
    level: int = logging.DEBUG
    format: str = LOG_FORMAT
    silence_spam: bool = True


#: The logging configuration of this process, if it has been created with :py:func:`setup_logging`
CONFIG: LoggingConfig | None = None


class MessageFilter(logging.Filter):
    """Exclude log records whose messages start with any of the given texts.

    This filter has to be attached to the logger that emits the record,
    since the filters of higher-level loggers are not applied to propagated records.
    This filter could be attached to a logging handler instead,
    but then the records would still be emitted to the other handlers.
    """

    def __init__(self, *texts: str):
        super().__init__()
        self.texts: tuple[str, ...] = texts

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(msg.startswith(text) for text in self.texts)


def silence_log_spam() -> None:
    """Silence the log spam of third-party libraries."""
    # "browser_proc" is the logger to which choreographer (used by Kaleido) forwards the stderr of the browser process.
    for logger_name in ["browser_proc", "choreographer", "kaleido", "logistro", "matplotlib"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    for logger_name in ["h5py", "numba", "Pillow", "PIL", "urllib3"]:
        logging.getLogger(logger_name).setLevel(logging.INFO)

    logging.getLogger("matplotlib.backends.backend_ps").addFilter(MessageFilter(
        "The PostScript backend does not support transparency"
    ))


def _apply_config(config: LoggingConfig) -> None:
    global CONFIG  # noqa: PLW0603
    logging.basicConfig(
        handlers=[
            logging.FileHandler(config.log_file_path),
            logging.StreamHandler()
        ],
        level=config.level,
        format=config.format
    )
    if config.silence_spam:
        silence_log_spam()
    CONFIG = config


def setup_logging(
        name: str = "pttools",
        log_dir: str | os.PathLike[str] | None = None,
        enable_faulthandler: bool = True,
        silence_spam: bool = True) -> None:
    """Configure logging to both file and console and optionally silence spam."""
    # Allow running this function only once for each process
    if not LOGGING_LOCK.acquire(blocking=False):
        return

    if enable_faulthandler and not faulthandler.is_enabled():
        faulthandler.enable()

    log_dir = Path(__file__).resolve().parent.parent / "logs" if log_dir is None else Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}.log"
    if log_file_path.exists():
        raise FileExistsError(f"The log file already exists, even though it should be per-process: {log_file_path}")
    _apply_config(LoggingConfig(log_file_path=log_file_path, silence_spam=silence_spam))


def setup_worker_logging(config: LoggingConfig | None) -> None:
    """Replicate the logging configuration of the main process in a worker process.

    This is meant to be used as the initializer of a :py:class:`concurrent.futures.ProcessPoolExecutor`,
    with the :py:data:`CONFIG` of the main process as the argument.
    With the "spawn" and "forkserver" start methods (the latter is the default on Linux since Python 3.14),
    the worker processes do not inherit the logging configuration of the main process,
    and without this the warnings emitted by the workers would be printed
    by the logging.lastResort handler without any formatting.

    :param config: The logging configuration of the main process, or None if it has not configured logging.
    """
    if config is None:
        return
    # If the worker process has been forked from the main process, it has inherited the logging configuration.
    if not LOGGING_LOCK.acquire(blocking=False):
        return
    _apply_config(config)
