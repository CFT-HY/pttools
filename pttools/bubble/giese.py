import logging
import os
import sys
import typing as tp

logger = logging.getLogger(__name__)

try:
    from giese.lisa import kappaNuMuModel
except ImportError:
    pttools_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parent_dir = os.path.dirname(pttools_path)
    msc_python_path = os.path.join(parent_dir, "msc-thesis2", "msc2-python")
    if os.path.isdir(msc_python_path):
        sys.path.append(msc_python_path)
        try:
            from giese.lisa import kappaNuMuModel
        except ImportError:
            logger.error("The msc-thesis2 repository was found but the Giese et al. code could not be imported.")
    kappaNuMuModel: tp.Optional[callable] = None
