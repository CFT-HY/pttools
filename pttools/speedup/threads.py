"""Utilities for measuring the scaling of the runtime with the number of Numba threads."""

import logging
from timeit import timeit
import typing as tp

import numba
import numpy as np

from pttools.speedup.options import NUMBA_DISABLE_JIT
import pttools.type_hints as th
from pttools.utils import AVAILABLE_CPU_CORES, powers_of_2

logger: logging.Logger = logging.getLogger(__name__)

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
