import numpy as np


def is_nan_or_none(value: float | None = None) -> bool:
    """Determine if a value is NaN or None"""
    return value is None or np.isnan(value)
