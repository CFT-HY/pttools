"""Format floats as scientific notation in LaTeX and Unicode"""

from fractions import Fraction
import math

UNICODE_MINUS: str = "\u2212"
UNICODE_SUPERSCRIPT: dict[int, int] = str.maketrans(
    "-+0123456789",
    "\u207b\u207a\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
)


def trim_zeros(s: str, strip_zeros: bool) -> str:
    """Trim zeros from the end of a number"""
    if strip_zeros and "." in s:
        s = s.rstrip("0").rstrip(".")
    return s or "0"


def decompose_float(
        x: float,
        precision: int,
        fixed_range: tuple[float, float],
        strip_zeros: bool) -> tuple[str | None, int | None]:
    """Split x into (mantissa_str, exponent).

    exponent is None when the value should be printed as a plain decimal.
    Returns (None, None) for non-finite input.
    """
    if not math.isfinite(x):
        return None, None
    if x == 0.0:  # also normalizes -0.0
        return "0", None

    mant_s, exp_s = f"{x:.{precision}e}".split("e")
    exp = int(exp_s)

    lo, hi = fixed_range
    if lo <= exp <= hi:
        # Same number of significant digits, but written positionally
        return trim_zeros(f"{x:.{max(precision - exp, 0)}f}", strip_zeros), None

    return trim_zeros(mant_s, strip_zeros), exp


def as_latex(
        x: float | Fraction | None,
        precision: int = 3,
        fixed_range: tuple[int, int] = (-3, 3),
        mul: str = r"\cdot",
        math_mode: bool = False,
        omit_unit_mantissa: bool = True,
        strip_zeros: bool = True) -> str:
    r"""Render x as LaTeX, e.g. 1.6e-12 -> r'1.6 \cdot 10^{-12}'.

    :param x: Input value
    :param precision: Significant digits after the leading one
    :param fixed_range: Exponent range printed positionally
    :param mul: Multiplication symbol
    :param math_mode: Wrap the result in $...$
    :param omit_unit_mantissa: Drop a mantissa of exactly 1
    :param strip_zeros: Drop zeros from the end of the mantissa
    """
    if x is None:
        return r"\mathrm{None}"
    if isinstance(x, Fraction):
        return str(x)
    mant, exp = decompose_float(x, precision, fixed_range, strip_zeros)

    if mant is None:
        body = r"\mathrm{NaN}" if math.isnan(v) else (r"\infty" if v > 0 else r"-\infty")
    elif exp is None:
        body = mant
    elif omit_unit_mantissa and mant in ("1", "-1"):
        body = f"{mant[:-1]}10^{{{exp}}}"
    else:
        body = f"{mant} {mul} 10^{{{exp}}}"

    return f"${body}$" if math_mode else body


def as_unicode(
        x: float | Fraction | None,
        precision: int = 3,
        fixed_range: tuple[int, int] = (-3, 3),
        minus: str = UNICODE_MINUS,
        mul: str = "\u00d7",
        omit_unit_mantissa: bool = True,
        strip_zeros: bool = True) -> str:
    """Render x as a Unicode string, e.g. 1.6e-12 -> '1.6 \u00d7 10\u207b\u00b9\u00b2'.

    :param x: Input value
    :param precision: Significant digits after the leading one
    :param fixed_range: Exponent range printed positionally
    :param minus: Sign used for the mantissa: '\u2212' (U+2212) is typographically correct and the default.
        Replace with the ASCII '-' if it doesn't render properly.
        The exponent always uses U+207B, since that is the only superscript minus available.
    :param mul: Multiplication symbol
    :param omit_unit_mantissa: Drop a mantissa of exactly 1
    :param strip_zeros: Drop zeros from the end of the mantissa
    """
    if x is None:
        return "None"
    if isinstance(x, Fraction):
        return str(x)

    mant, exp = decompose_float(x, precision, fixed_range, strip_zeros)

    if mant is None:
        v = float(x)
        return "NaN" if math.isnan(v) else ("\u221e" if v > 0 else f"{minus}\u221e")

    mant = mant.replace("-", minus)

    if exp is None:
        return mant

    sup = str(exp).translate(UNICODE_SUPERSCRIPT)
    if omit_unit_mantissa and mant in ("1", f"{minus}1"):
        return f"{mant[:-1]}10{sup}"
    return f"{mant} {mul} 10{sup}"


if __name__ == "__main__":
    vals = [1.602176634e-19, -3.14159e12, 6.674e-11, 1e-12, -1e-12,
            0.0, -0.0, 42.0, 0.00123, 9.999e-13, 1.5, float("nan"),
            float("inf"), float("-inf")]
    for v in vals:
        print(f"{v!r:>24}   {as_latex(v):<28} {as_unicode(v)}")
