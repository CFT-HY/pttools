import decimal
import typing as tp

import colorama
import numpy as np

from . import math as test_math

DEFAULT_FMT = ".8e"
HIGH_PREC = 10

colorama.init()


def row_to_str(row: np.ndarray, close: np.ndarray, fmt: str = DEFAULT_FMT) -> str:
    lst = [
        f"{'' if ok else colorama.Fore.RED}{act:{fmt}}{''if ok else colorama.Fore.RESET}"
        for act, ok in zip(row, close)
    ]
    return f"[{', '.join(lst)}]"


def pairs_to_rows(actual: np.ndarray, desired: np.ndarray, close: np.ndarray, fmt: str = DEFAULT_FMT) -> tp.List[str]:
    return [
        f"{'' if ok else colorama.Fore.RED}"
        f"{act:{fmt}}, {des:{fmt}}, {test_math.rel_diff_scalar(act, des):{fmt}}, {act - des:{fmt}}"
        f"{'' if ok else colorama.Fore.RESET}"
        for act, des, ok in zip(actual, desired, close)
    ]


def print_1d_small(actual: np.ndarray, desired: np.ndarray, close: np.ndarray, fmt: str = DEFAULT_FMT):
    print("actual:")
    print(row_to_str(actual, close, fmt))
    print("desired:")
    print(row_to_str(desired, close, fmt))
    print("rdiff:")
    print(row_to_str(test_math.rel_diff_arr(actual, desired), close, fmt))
    print("adiff:")
    print(row_to_str(actual - desired, close, fmt))


def print_1d_large(actual: np.ndarray, desired: np.ndarray, close: np.ndarray, fmt: str = DEFAULT_FMT):
    print("actual          desired         rdiff           adiff")
    print("\n".join(pairs_to_rows(actual, desired, close, fmt)))


def print_1d(actual: np.ndarray, desired: np.ndarray, close: np.ndarray):
    if actual.size < 10:
        print_1d_small(actual, desired, close)
    else:
        print_1d_large(actual, desired, close)


def print_2d(arr: np.ndarray, close: np.ndarray, fmt: str = DEFAULT_FMT):
    rows = "\n ".join([row_to_str(row, ok, fmt) for row, ok in zip(arr, close)])
    print(f"[{rows}]")


def print_full_prec(x: float):
    print(decimal.Decimal(x))


def high_prec_float_str(x: float) -> str:
    return f"{x:.{HIGH_PREC}g}"


def print_high_prec(x: tp.Union[float, np.ndarray]):
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
