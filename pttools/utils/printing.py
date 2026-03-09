"""Printing utilities for testing"""

import decimal

import numpy as np

import pttools.type_hints as th
from pttools.utils.math import rel_diff_arr, rel_diff_scalar

DEFAULT_FMT = ".8e"
HIGH_PREC = 10

RED: str
RESET: str
try:
    import colorama
    colorama.init()
    RED = colorama.Fore.RED
    RESET = colorama.Fore.RESET
except ModuleNotFoundError:
    RED = ""
    RESET = ""


def row_to_str(row: th.FloatArr1D, close: th.BoolArr1D, fmt: str = DEFAULT_FMT) -> str:
    """Convert an array row to string with color"""
    lst = [
        f"{'' if ok else RED}{act:{fmt}}{''if ok else RESET}"
        for act, ok in zip(row, close)
    ]
    return f"[{', '.join(lst)}]"


def pairs_to_rows(
        actual: th.FloatArr1D,
        desired: th.FloatArr1D,
        close: th.BoolArr1D, fmt: str = DEFAULT_FMT) -> list[str]:
    """Convert pairs of actual and desired values to string rows with color"""
    return [
        f"{'' if ok else RED}"
        f"{act:{fmt}}, {des:{fmt}}, {rel_diff_scalar(act, des):{fmt}}, {act - des:{fmt}}"
        f"{'' if ok else RESET}"
        for act, des, ok in zip(actual, desired, close)
    ]


def print_1d_small(actual: th.FloatArr1D, desired: th.FloatArr1D, close: th.BoolArr1D, fmt: str = DEFAULT_FMT) -> None:
    """Print a small 1D array"""
    print("actual:")
    print(row_to_str(actual, close, fmt))
    print("desired:")
    print(row_to_str(desired, close, fmt))
    print("rdiff:")
    print(row_to_str(rel_diff_arr(actual, desired), close, fmt))
    print("adiff:")
    print(row_to_str(actual - desired, close, fmt))


def print_1d_large(actual: th.FloatArr1D, desired: th.FloatArr1D, close: th.BoolArr1D, fmt: str = DEFAULT_FMT) -> None:
    """Print a large 1D array"""
    print("actual          desired         rdiff           adiff")
    print("\n".join(pairs_to_rows(actual, desired, close, fmt)))


def print_1d(actual: th.FloatArr1D, desired: th.FloatArr1D, close: th.BoolArr1D) -> None:
    """Print a 1D array"""
    if actual.size < 10:
        print_1d_small(actual, desired, close)
    else:
        print_1d_large(actual, desired, close)


def print_2d(arr: th.FloatArr2D, close: th.BoolArr2D, fmt: str = DEFAULT_FMT) -> None:
    """Print a 2D array"""
    rows = "\n ".join([row_to_str(row, ok, fmt) for row, ok in zip(arr, close)])
    print(f"[{rows}]")


def print_full_prec(x: float) -> None:
    """Print a float with full precision"""
    print(decimal.Decimal(x))


def high_prec_float_str(x: float) -> str:
    """Convert a float to a string with high precision"""
    return f"{x:.{HIGH_PREC}g}"


def print_high_prec(x: th.FloatOrArr) -> None:
    """Print a value or an array with high precision"""
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            print("[" + ", ".join([high_prec_float_str(elem) for elem in x]) + "]")
        if x.ndim == 2:
            print("[" + "\n".join([", ".join([str(high_prec_float_str(elem) for elem in line)]) for line in x]) + "]")
        # These other ways tend to result in extra spaces between the elements
        # with np.printoptions(precision=10, edgeitems=30, linewidth=1000):
        #     print(x)
        # print(np.array2string(
        #     x,
        #     edgeitems=30,
        #     floatmode="maxprec",
        #     max_line_width=100,
        #     precision=PRINT_PRECISION,
        #     separator=", ",
        # ))
        # print(np.array_repr(x, precision=PRINT_PRECISION))
        # print(np.array_str(x, precision=PRINT_PRECISION))
    else:
        print(high_prec_float_str(x))
