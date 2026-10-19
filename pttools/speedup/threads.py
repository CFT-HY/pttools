"""Utilities for measuring the scaling of the runtime with the number of Numba threads."""

import logging
import math
from timeit import timeit
import typing as tp

import numba
import numpy as np

from pttools.speedup.options import NUMBA_DISABLE_JIT
import pttools.type_hints as th
from pttools.utils import AVAILABLE_CPU_CORES

logger: logging.Logger = logging.getLogger(__name__)


# This is defined before the constants, as they use it.
def powers_of_2(max_val: int, start_exp: int = 0, min_end_exp: int = 0, include_max: bool = False) -> th.IntArr1D:
    """Get the powers of 2 up to a certain value."""
    ret = [2 ** i for i in range(start_exp, max(min_end_exp, int(math.log2(max_val)) + 1))]
    # If max_val is not a power of 2
    if include_max and max_val.bit_count() != 1:
        ret.append(max_val)
    return np.array(ret)


#: Default numbers of threads for performance comparisons
DEFAULT_VARYING_NUMBA_THREADS: th.IntArr1D = powers_of_2(
    1 if AVAILABLE_CPU_CORES is None else AVAILABLE_CPU_CORES)


def time_and_log(
        name: str,
        stmt: str,
        setup: str,
        n_iterations: int,
        n_threads: int,
        log: bool = True,
        file: tp.TextIO | None = None) -> float:
    """Measure the runtime of a statement with :func:`timeit.timeit` and log the result.

    :param name: name of the measurement for the log message
    :param stmt: the statement to be timed
    :param setup: the setup statement, which is run once before the timing
    :param n_iterations: number of times to run the statement
    :param n_threads: number of threads for the log message. This does not set the number of threads.
    :param log: whether to log the result
    :param file: file to which the log message is also written
    :return: total runtime in seconds
    """
    result = timeit(stmt=stmt, setup=setup, number=n_iterations)
    # layer = numba.threading_layer()
    if log:
        text = f"{name} performance with {n_threads} threads and {n_iterations} iterations: " \
               f"{result:.2f} s, {result / n_iterations:.6f} s/iteration"
        logger.info(text)
        if file is not None:
            file.write(f"{text}\n")
    return result


def time_with_varying_numba_threads(
        name: str,
        stmt: str,
        setup: str,
        n_iterations: int,
        n_threads: th.IntArr1D = DEFAULT_VARYING_NUMBA_THREADS,
        log: bool = True,
        file: tp.TextIO | None = None) -> tuple[th.IntArr1D, th.FloatArr1D]:
    """Measure the runtime of a statement with different numbers of Numba threads.

    If Numba JIT compilation is disabled, the runtime is measured only once, with one thread.
    The original number of Numba threads is restored afterwards.

    :param name: name of the measurement for the log messages
    :param stmt: the statement to be timed
    :param setup: the setup statement, which is run once before each timing
    :param n_iterations: number of times to run the statement for each number of threads
    :param n_threads: numbers of threads to be tested
    :param log: whether to log the results
    :param file: file to which the log messages are also written
    :return: the numbers of threads and the corresponding total runtimes in seconds
    """
    if NUMBA_DISABLE_JIT:
        return np.array([1]), np.array([time_and_log(name, stmt, setup, n_iterations, 1, log, file)])

    default_threads = numba.get_num_threads()
    times = []
    try:
        for n_np in n_threads:
            n = int(n_np)
            numba.set_num_threads(n)
            times.append(time_and_log(name, stmt, setup, n_iterations, n, log, file))
    finally:
        numba.set_num_threads(default_threads)

    if log:
        text = f"Numba threading layer: {numba.threading_layer()}, CPU cores: {AVAILABLE_CPU_CORES}"
        logger.debug(text)
        if file is not None:
            file.write(f"{text}\n")
    return n_threads, np.array(times)
