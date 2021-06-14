"""Common utilities for ssm_compare.py and ssm_paper_utils.py"""

import enum
import logging

import numpy as np

logger = logging.getLogger(__name__)


@enum.unique
class Method(str, enum.Enum):
    E_CONSERVING = "e_conserving"


@enum.unique
class PSType(str, enum.Enum):
    GW = "gw"
    V = "v"
    UNKNOWN = ""


@enum.unique
class Position(str, enum.Enum):
    HIGH = "high"
    LOW = "low"
    MED = "med"


@enum.unique
class Strength(str, enum.Enum):
    INTER = "inter"
    STRONG = "strong"
    WEAK = "weak"


def get_ymax_location(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Returns x, y coordinates of maximum of array y"""
    ymax = max(y)
    xmax = x[np.where(y == ymax)][0]
    return np.array([xmax, ymax])
