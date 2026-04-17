import logging
import typing as tp

from kaleido._kaleido_tab import KaleidoError

logger = logging.getLogger(__name__)


def plotly_fix(func: tp.Callable) -> tp.Callable:
    """Suppress Kaleido plotting failures for e.g. unit tests headless machines"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KaleidoError as err:
            logger.exception(err)
    return wrapper
