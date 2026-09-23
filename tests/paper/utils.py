"""Common utilities for ssm_compare.py and ssm_paper_utils.py."""

import enum
import logging

import numpy as np

import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)


@enum.unique
class PSType(enum.StrEnum):
    GW = "gw"
    V = "v"
    UNKNOWN = ""


@enum.unique
class Position(enum.StrEnum):
    HIGH = "high"
    LOW = "low"
    MED = "med"


@enum.unique
class Strength(enum.StrEnum):
    INTER = "inter"
    STRONG = "strong"
    WEAK = "weak"


def get_ymax_location(x: th.FloatArr1D, y: th.FloatArr1D) -> th.FloatArr1D:
    """Returns x, y coordinates of maximum of array y."""
    ymax = np.max(y)
    xmax = x[np.where(y == ymax)][0]
    return np.array([xmax, ymax])
