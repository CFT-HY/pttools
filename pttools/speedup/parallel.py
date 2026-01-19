"""Utilities for parallel execution of functions using multiple Python processes with concurrent.futures"""

import concurrent.futures as cf
# import datetime
import multiprocessing
from concurrent.futures.process import BrokenProcessPool
import logging
import os
import platform
import subprocess
import time
import typing as tp

import numpy as np
try:
    import psutil
except ModuleNotFoundError:
    psutil = None

from pttools.speedup.options import IS_LINUX, MAX_WORKERS_DEFAULT, UNAME, START_METHOD

logger = logging.getLogger(__name__)


class FakeFuture:
    """A fake future object for single-threaded execution"""
    # pylint: disable=too-few-public-methods

    def __init__(self, func: tp.Callable, *args, **kwargs):
        self._result = func(*args, **kwargs)

    def result(self):
        """Get the result of the function execution"""
        return self._result


class LoggingRunner:
    """A handler for logging the execution status of a function that is run in parallel"""
    def __init__(
            self,
            func: tp.Callable,
            arr_size: int,
            unpack_params: bool,
            args: tuple | list = (),
            kwargs: dict[str, tp.Any] | None = None,
            log_progress_element: int | None = None,
            log_progress_percentage: float | None = None):
        self.func = func
        self.arr_size = arr_size
        self.unpack_params = unpack_params
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
        self.log_progress_element = log_progress_element
        self.log_progress_percentage = log_progress_percentage

    def run(self, param, index: int | None = None, multi_index: tp.Iterable | None = None):
        if self.unpack_params:
            ret = self.func(*param, *self.args, **self.kwargs)
        else:
            ret = self.func(param, *self.args, **self.kwargs)

        if index is not None:
            percentage = index / self.arr_size * 100
            percentage_prev = (index - 1) / self.arr_size * 100
            if (self.log_progress_element is not None and index % self.log_progress_element == 0) \
                or (self.log_progress_percentage is not None and
                    np.floor(percentage / self.log_progress_percentage) != np.floor(
                    percentage_prev / self.log_progress_percentage)):
                if multi_index is None:
                    logger.debug("Processed item %d/%d, %.2f %%", index, self.arr_size, percentage)
                else:
                    logger.debug(
                        "Processed item %s, %d/%d, %.2f %%",
                        multi_index, index, self.arr_size, percentage
                    )

        return ret


def parallel_debug_message(
        info: str = None,
        err: Exception = None,
        max_workers: int = None,
        single_thread: bool = None,
        start_time: float = None,
        n_dmesg_lines: int = 100) -> str:
    end_time = time.perf_counter()
    msg = info
    if info is not None and info[-1] != " ":
        msg += " "
    if err is not None:
        msg += f"Exception arguments: {err.args}. "
    if start_time is not None:
        msg += f"Executor runtime: {end_time - start_time:.2f} s. "
    msg += (
        f"OS: {UNAME.system} ({UNAME.release}), CPU: {UNAME.processor} ({UNAME.machine}), "
        f"Python: {platform.python_version()}, "
        f"Start method: {START_METHOD}, available: {multiprocessing.get_all_start_methods()}"
    )
    if max_workers is not None:
        msg += f", max_workers={max_workers}"
    if single_thread is not None:
        msg += f", single_thread={single_thread}"

    if psutil is None:
        msg += ". Please install psutil for more info."
    else:
        cpu = psutil.getloadavg()
        cpu_count = psutil.cpu_count()
        ram = psutil.virtual_memory()
        msg += (
            f". CPU cores: {cpu_count}, CPU use: "
            f"1 min {cpu[0] / cpu_count * 100} %, "
            f"5 min {cpu[1] / cpu_count * 100} %, "
            f"15 min {cpu[2] / cpu_count * 100} %. "
            f"RAM use: {ram.used * 1e-9:.2f} / {ram.total * 1e-9:.2f} GB = {ram.percent} %, "
            f"available {ram.available} GB."
        )
        if ram.percent > 80:
            msg += (
                " RAM use is high. "
                "Please reduce the number of worker processes or close applications running in the background."
            )
    if IS_LINUX:
        if os.geteuid() == 0:
            try:
                dmesg = subprocess.run(
                    ["dmesg", "|", "tail", "-n", str(n_dmesg_lines)],
                    capture_output=True,
                    stderr=subprocess.STDOUT
                )
                print(f"Last {n_dmesg_lines} lines from dmesg:")
                print(dmesg.stdout.decode("utf-8"))
            except Exception as e:
                msg += f" Failed to get dmesg for more info: {e}"
        else:
            msg += " Skipping dmesg printing since not running as root."
    return msg


def run_parallel(
        func: tp.Callable,
        params: np.ndarray,
        max_workers: int = MAX_WORKERS_DEFAULT,
        multiple_params: bool = False,
        unpack_params: bool = False,
        output_dtypes: tuple[tp.Type, ...] | list[tp.Type] | None = None,
        return_arr_shape: tuple[int, ...] | None = None,
        log_progress_element: int | None = None,
        log_progress_percentage: float | None = None,
        args: tp.Union[list, tuple] = (),
        kwargs: dict[str, tp.Any] | None = None,
        single_thread: bool = False) -> tp.Optional[tp.Union[np.ndarray, tuple[np.ndarray, ...]]]:
    """Run the given function with multiple parameters in parallel

    :param func: The function to be executed in parallel
    :param params: Array of the function parameters
    :param max_workers: Maximum number of worker processes
    :param multiple_params: Whether the last dimension of the parameter array contains multiple parameters for each function call
    :param unpack_params: Whether the multiple parameters should be unpacked before giving them to the function
    :param output_dtypes: If the function has multiple output values, their types should be given here
    :param return_arr_shape: Shape of the array given by func. If None, the function should return single values.
    :param log_progress_element: Log progress every n element
    :param log_progress_percentage: Log progress every x %
    :param args: common arguments for the function
    :param kwargs: common kwargs for the function
    :param single_thread: disable parallelism for debugging and profiling
    :return: Numpy arrays for each output value
    """
    if kwargs is None:
        kwargs = {}
    if max_workers is None:
        max_workers = MAX_WORKERS_DEFAULT

    flags = ["refs_ok"]
    arr_size: int
    if multiple_params:
        flags.append("reduce_ok")
        flags.append("external_loop")
        op_axes = [None, [*list(range(params.ndim-1)), -1]]
        arr_size = np.prod(params.shape[:-1])
    else:
        flags.append("c_index")
        flags.append("multi_index")
        op_axes = None
        arr_size = np.prod(params.shape)

    runner = LoggingRunner(
        func,
        arr_size=arr_size, unpack_params=unpack_params,
        args=args, kwargs=kwargs,
        log_progress_element=log_progress_element, log_progress_percentage=log_progress_percentage
    )

    start_time = time.perf_counter()
    # start_datetime = datetime.datetime.now().astimezone().isoformat()
    try:
        with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
            # Submit parallel execution
            with np.nditer(
                    [params, None],
                    flags=flags,
                    op_flags=[["readonly"], ["readwrite", "allocate"]],
                    op_axes=op_axes,
                    op_dtypes=[params.dtype, object],
                    order="C") as it:
                if single_thread:
                    for ind, (param, fut) in enumerate(it):
                        multi_index = None if multiple_params else it.multi_index
                        fut[...] = FakeFuture(runner.run, param, index=ind, multi_index=multi_index)
                else:
                    for ind, (param, fut) in enumerate(it):
                        multi_index = None if multiple_params else it.multi_index
                        fut[...] = ex.submit(runner.run, param, index=ind, multi_index=multi_index)
                futs = it.operands[1]

            # Collect results

            # No output
            # if output_dtypes is None:
            #     return None

            # Array output
            if return_arr_shape is not None:
                output_arr = np.empty((*futs.shape, *return_arr_shape), dtype=output_dtypes[0])
                # axes = list(range(futs.ndim)) # + list((-1, ) * len(return_arr_shape))
                # op_axes = [axes, axes]
                with np.nditer(
                        futs,
                        flags=("refs_ok", "c_index", "multi_index"),
                        # op_flags=[["readonly"], ["writeonly"]],
                        # op_axes=op_axes,
                        order="C") as it:
                    for fut in it:
                        output_arr[*it.multi_index, :] = fut.item().result()
                return output_arr

            # Single output
            single_output = False
            if output_dtypes is None:
                single_output = True
                output_arr = None
            elif len(output_dtypes) == 1:
                single_output = True
                output_arr = np.empty_like(futs, dtype=output_dtypes[0])

            if single_output:
                with np.nditer(
                        [futs, output_arr],
                        flags=("refs_ok", "c_index", "multi_index"),
                        # op_flags=[["readonly"], ["writeonly"]],
                        order="C") as it:
                    for fut, res in it:
                        res[...] = fut.item().result()
                    return it.operands[1]

            # Multiple outputs
            op_flags2 = [["readonly"], *[["writeonly"]] * len(output_dtypes)]
            output_arrs = tuple(np.empty(futs.shape, dtype=dtype) for dtype in output_dtypes)
            with np.nditer(
                    [futs, *output_arrs],
                    flags=("refs_ok", "c_index", "multi_index"),
                    op_flags=op_flags2,
                    order="C") as it:
                for elems in it:
                    res = elems[0].item().result()
                    try:
                        for arr, val in zip(elems[1:], res):
                            arr[...] = val
                    except ValueError as e:
                        logger.exception("Could not store result to output array. Got: %s", res, exc_info=e)
                        raise e
            return output_arrs
    except BrokenProcessPool as err:
        msg = parallel_debug_message(
            info="Parallel execution failed due to a system error. ",
            err=err,
            max_workers=max_workers,
            single_thread=single_thread,
            start_time=start_time
        )
        raise BrokenProcessPool(msg) from err


# This seems to be fixed as of 2025.
# See this documentation for the number of CPU cores available on the GitHub Actions runners:
# https://docs.github.com/en/actions/using-github-hosted-runners/using-github-hosted-runners/about-github-hosted-runners#standard-github-hosted-runners-for-public-repositories
# if GITHUB_ACTIONS and os.name == "nt":
#     # This value is based on the output of Numba sysinfo on the GitHub Actions Windows runner
#     num_threads = 4
#     logger.warning(
#         "Detected GitHub Actions Windows runner with %s threads. "
#         "Setting the number of threads to %s to work around a Numba bug in detecting the number of CPUs.",
#         get_num_threads(), num_threads
#     )
#     set_num_threads(num_threads)
