"""Logging configuration"""

import faulthandler
import logging
import os
import threading
import time

logging_lock = threading.Lock()


def setup_logging(log_dir: str = None, enable_faulthandler: bool = True, silence_spam: bool = True):
    """Configure logging to both file and console and optionally silence spam"""
    # Allow running this function only once for each process
    if not logging_lock.acquire(blocking=False):
        return

    if enable_faulthandler and not faulthandler.is_enabled():
        faulthandler.enable()

    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"tests_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}.txt")
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
        logging.getLogger("matplotlib").setLevel(logging.INFO)
        logging.getLogger("numba").setLevel(logging.INFO)
        logging.getLogger("Pillow").setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)
        logging.getLogger("urllib3").setLevel(logging.INFO)
