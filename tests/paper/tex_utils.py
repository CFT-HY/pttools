"""LaTeX utilities.

Modified from
`sound-shell-model/paper/python/tex_utils.py
<https://bitbucket.org/hindmars/sound-shell-model/src/master/paper/python/tex_utils.py>`_.
"""

import numpy as np

from tests.paper import sigfig as sf


def tex_sf(x: float, n: int = 2, sci_notn_threshold: int = 2, mult: str = "\\times") -> str:
    """Format x to n significant figures as LaTeX math, in scientific notation for large or small exponents.

    Scientific notation is used if the absolute value of the exponent exceeds sci_notn_threshold.
    """
    expon = np.ceil(np.log10(abs(x))) - 1
    if np.abs(expon) > sci_notn_threshold:
        manti = x * 10**(-expon)
        manti_str = sf.round_sig(manti, n)
        if manti_str == "10":
            return f"$ 10^{{{expon+1:.0f}}} $"
        return f"$ {sf.round_sig(manti, n)}{mult} 10^{{{expon:.0f}}} $"
    return f"$ {sf.round_sig(x, n)} $"


def tex_sf_signed(x: float, n: int, sci_notn_threshold: int = 2, mult: str = "\\times") -> str:
    """Format x to n significant figures with an explicit sign as LaTeX math, like :func:`tex_sf`."""
    expon = np.ceil(np.log10(abs(x))) - 1
    if np.abs(expon) > sci_notn_threshold:
        manti = x*10**(-expon)
        manti_str = sf.round_sig(manti, n)
        if manti_str == "10":
            return f"$ {sf.round_sig_signed(manti, n)}{mult} 10^{{{expon:.0f}}} $"
        return f"$ {sf.round_sig_signed(manti, n)}{mult} 10^{{{expon:.0f}}} $"
    return f"$ {sf.round_sig_signed(x, n)} $"
