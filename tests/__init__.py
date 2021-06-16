import logging
import os
import time

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

log_file_path = os.path.join(log_dir, f"tests_{time.strftime('%Y-%m-%d_%H-%M-%S')}_{os.getpid()}")

logging.basicConfig(
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ],
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)-8s %(module)-16s %(funcName)-32s %(lineno)-4d %(message)s'
)
