# import logging
import typing as tp

# logger = logging.getLogger(__name__)

try:
    # logger.info("Giese code imported successfully.")
    from giese.lisa import kappaNuMuModel
except ImportError:
    # logger.info("Giese could not be imported.")
    kappaNuMuModel: tp.Optional[callable] = None
