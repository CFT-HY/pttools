import logging
import os
import time

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
    format='%(asctime)s %(levelname)-8s %(module)-16s %(funcName)-32s %(lineno)-4d %(message)s'
)
# Disable log spam
logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
