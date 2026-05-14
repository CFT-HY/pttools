import logging
import typing as tp

from kaleido._kaleido_tab import KaleidoError

logger = logging.getLogger(__name__)


def plotly_fix(func: tp.Callable) -> tp.Callable:
    """Suppress Kaleido plotting failures

    The Kaleido library Plotly uses to create raster graphics such as PNG
    does not work on all headless machines.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KaleidoError as err:
            logger.exception(err)
    return wrapper
