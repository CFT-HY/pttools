"""Logging configuration"""

import faulthandler
import logging
import os
import time


def setup_logging(log_dir: str = None, enable_faulthandler: bool = True, silence_spam: bool = True):
    if enable_faulthandler and not faulthandler.is_enabled():
        faulthandler.enable()

    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"tests_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}.txt")
    logging.basicConfig(
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()
        ],
        level=logging.DEBUG,
        # level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(module)-20s %(funcName)-32s %(lineno)-4d %(message)s'
    )
    if silence_spam:
        logging.getLogger("matplotlib").setLevel(logging.INFO)
        logging.getLogger("numba").setLevel(logging.INFO)
        logging.getLogger("Pillow").setLevel(logging.INFO)
