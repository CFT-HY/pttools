"""Utility functions and constants for unit testing"""

import decimal
import os.path
import typing as tp

import numpy as np

TEST_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(TEST_PATH, "test_data")
TEST_RESULT_PATH = os.path.join(os.path.dirname(TEST_PATH), "test-results")
TEST_FIGURE_PATH = os.path.join(TEST_RESULT_PATH, "figures")
PRINT_PRECISION = 10


def print_full_prec(x: float):
    print(decimal.Decimal(x))


def high_prec_float_str(x: float) -> str:
    return f"{x:.{PRINT_PRECISION}g}"


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
