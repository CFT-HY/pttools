"""Logging configuration"""

import faulthandler
import logging
import os
from threading import Lock
import time

LOGGING_LOCK = Lock()


class MessageFilter(logging.Filter):
    """Exclude log records whose messages start with any of the given texts

    This filter has to be attached to the logger that emits the record,
    since the filters of higher-level loggers are not applied to propagated records.
    This filter could be attached to a logging handler instead,
    but then the records would still be emitted to the other handlers.
    """
    def __init__(self, *texts: str):
        super().__init__()
        self.texts = texts

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(msg.startswith(text) for text in self.texts)


def setup_logging(
        name: str = "pttools",
        log_dir: str | None = None,
        enable_faulthandler: bool = True,
        silence_spam: bool = True):
    """Configure logging to both file and console and optionally silence spam"""
    # Allow running this function only once for each process
    if not LOGGING_LOCK.acquire(blocking=False):  # pylint: disable=consider-using-with
        return

    if enable_faulthandler and not faulthandler.is_enabled():
        faulthandler.enable()

    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}.log")
    if os.path.exists(log_file_path):
        raise FileExistsError(f"The log file already exists, even though it should be per-process: {log_file_path}")
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ],
        level=logging.DEBUG,
        # level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(module)-20s %(funcName)-32s %(lineno)-4d %(process)-3d %(message)s'
    )
    if silence_spam:
        for name in ["choreographer", "kaleido", "logistro", "matplotlib"]:
            logging.getLogger(name).setLevel(logging.WARNING)
        for name in ["h5py", "numba", "Pillow", "PIL", "urllib3"]:
            logging.getLogger(name).setLevel(logging.INFO)

        logging.getLogger("matplotlib.backends.backend_ps").addFilter(MessageFilter(
            "The PostScript backend does not support transparency"
        ))
