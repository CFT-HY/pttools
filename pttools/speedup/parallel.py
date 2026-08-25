"""Utilities for parallel execution of functions using multiple Python processes with concurrent.futures"""

import atexit
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
# import datetime
import logging
from multiprocessing import set_forkserver_preload
import sys
from threading import Lock
import time
import typing as tp

import numpy as np
from numpy.typing import NDArray

from pttools.speedup.options import MAX_WORKERS_DEFAULT
from pttools.utils import SUPPORTS_FREETHREADING, SUPPORTS_INTERPRETER_POOL

try:
    from concurrent.futures import InterpreterPoolExecutor  # type: ignore[attr-defined]
except ImportError:
    class InterpreterPoolExecutor:  # type: ignore[no-redef]
        def __enter__(self):
            raise NotImplementedError("InterpreterPoolExecutor is only available in Python 3.14 and later.")

        def __init__(self, *args, **kwargs):
            raise NotImplementedError("InterpreterPoolExecutor is only available in Python 3.14 and later.")

logger = logging.getLogger(__name__)

POOL: ProcessPoolExecutor | None = None
POOL_LOCK: Lock = Lock()

# On Linux, forkserver is the default start method since Python 3.14.
# On Windows and macOS, the default start method is spawn, which is slow.
# This config preloads the target libraries in the forkserver process before forking it,
# to speed up the start of new processes.
DEFAULT_FORKSERVER_PRELOAD: list[str] = [
    "numba", "numpy", "scipy",
    "pttools.analysis", "pttools.bubble", "pttools.models",
    "pttools.omgw0", "pttools.speedup", "pttools.ssm", "pttools.utils",
    "pttools.logging", "pttools.type_hints"
]
set_forkserver_preload(DEFAULT_FORKSERVER_PRELOAD)


class FakeExecutor:
    """A fake executor for single-threaded execution"""
    # pylint: disable=too-few-public-methods

    @staticmethod
    def submit(func: tp.Callable, *args, **kwargs) -> "FakeFuture":
        """Submit a function for execution and return a future object"""
        return FakeFuture(func, *args, **kwargs)


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


def get_global_process_pool(max_workers: int = MAX_WORKERS_DEFAULT) -> ProcessPoolExecutor:
    global POOL
    with POOL_LOCK:
        if POOL is None:
            POOL = ProcessPoolExecutor(max_workers=max_workers)
            atexit.register(POOL.shutdown)
        return POOL


@contextmanager
def get_process_pool(
        max_workers: int = MAX_WORKERS_DEFAULT,
        single_thread: bool = False,
        global_pool: bool = False) -> tp.Iterator[FakeExecutor | ProcessPoolExecutor]:
    """Get a process pool for parallel execution.

    :param max_workers: Maximum number of worker processes
    :param single_thread: Whether to disable parallelism for debugging and profiling
    :param global_pool: Whether to use a global process pool for parallel execution.
        This avoids the overhead of creating a new process pool for each parallel execution.
    :return: The pool executor
    """
    if single_thread:
        yield FakeExecutor()
    elif global_pool:
        yield get_global_process_pool(max_workers=max_workers)
    else:
        yield ProcessPoolExecutor(max_workers=max_workers)


@contextmanager
def get_pool(
        max_workers: int = MAX_WORKERS_DEFAULT,
        single_thread: bool = False,
        global_pool: bool = False) \
        -> tp.Iterator[FakeExecutor | InterpreterPoolExecutor | ProcessPoolExecutor | ThreadPoolExecutor]:
    """Get a pool for parallel execution

    Returns a ThreadPoolExecutor if free-threading is supported and the GIL is disabled.
    Otherwise, returns an InterpreterPoolExecutor if supported, or a ProcessPoolExecutor otherwise.

    This is not yet used by the rest of PTtools, since free-threading and InterpreterPoolExecutor are experimental.

    :param max_workers: Maximum number of worker processes or threads
    :param single_thread: Whether to disable parallelism for debugging and profiling
    :param global_pool: Whether to use a global process pool for parallel execution.
        This avoids the overhead of creating a new process pool for each parallel execution.
    :return: The pool executor
    """
    if single_thread:
        yield FakeExecutor()
    elif SUPPORTS_FREETHREADING and not sys._is_gil_enabled():  # type: ignore[attr-defined]
        yield ThreadPoolExecutor(max_workers=max_workers)
    elif SUPPORTS_INTERPRETER_POOL:
        yield InterpreterPoolExecutor(max_workers=max_workers)
    elif global_pool:
        yield get_global_process_pool(max_workers=max_workers)
    else:
        yield ProcessPoolExecutor(max_workers=max_workers)


def parallel_debug_message(
        info: str | None = None,
        err: Exception | None = None,
        max_workers: int | None = None,
        single_thread: bool | None = None,
        start_time: float | None = None,
        kwargs: dict[str, tp.Any] | None = None) -> str:
    end_time = time.perf_counter()
    if info is None:
        msg = ""
    elif info[-1] == " ":
        msg = info
    else:
        msg = f"{info} "

    if err is not None:
        msg += f"Exception arguments: {err.args}. "
    if start_time is not None:
        msg += f"Executor runtime: {end_time - start_time:.2f} s. "

    if max_workers is not None:
        msg += f", max_workers={max_workers}"
    if single_thread is not None:
        msg += f", single_thread={single_thread}"

    if kwargs:
        msg += ", ".join([f"{name}={value}" for name, value in kwargs.items()])

    return msg


def log_parallel_ready(
        executor: Executor | FakeExecutor,
        n_workers: int,
        n_tasks: int,
        start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    cpu_time_per_task = elapsed * n_workers / n_tasks
    logger.info(
        "Parallel processing ready. Executor: %s, workers: %s, tasks: %s, time: %.3f s, thread time / task: %.3f s",
        executor, n_workers, n_tasks, elapsed, cpu_time_per_task
    )


def run_parallel(
        func: tp.Callable,
        params: NDArray,
        max_workers: int = MAX_WORKERS_DEFAULT,
        multiple_params: bool = False,
        unpack_params: bool = False,
        output_dtypes: tuple[tp.Type, ...] | list[tp.Type] | None = None,
        return_arr_shape: tuple[int, ...] | None = None,
        log_progress_element: int | None = None,
        log_progress_percentage: float | None = None,
        log_start_finish: bool = True,
        args: list | tuple = (),
        kwargs: dict[str, tp.Any] | None = None,
        single_thread: bool = False,
        global_pool: bool = False) -> NDArray | tuple[NDArray, ...] | None:
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
    :param log_start_finish: Log start and finish of parallel processing
    :param args: common arguments for the function
    :param kwargs: common kwargs for the function
    :param single_thread: disable parallelism for debugging and profiling
    :param global_pool: Whether to use a global process pool for parallel execution.
        This removes the overhead of recreating the process pool,
        but updates in the global state (such as caches)
        after the pool is first created will not be updated to the subprocesses.
    :return: Numpy arrays for each output value
    """
    if kwargs is None:
        kwargs = {}
    if max_workers is None:
        raise ValueError(f"Got invalid max_workers: {max_workers}")

    n_tasks = int(np.prod(params.shape[:-1])) if multiple_params else params.size
    n_workers = 1 if single_thread else max_workers

    flags: list[tp.Literal["c_index", "external_loop", "multi_index", "reduce_ok", "refs_ok"]] = ["refs_ok"]
    arr_size: int
    if multiple_params:
        flags.append("reduce_ok")
        flags.append("external_loop")
        op_axes = [None, [*list(range(params.ndim-1)), -1]]
        arr_size = int(np.prod(params.shape[:-1]))
    else:
        flags.append("c_index")
        flags.append("multi_index")
        op_axes = None
        arr_size = int(np.prod(params.shape))

    runner = LoggingRunner(
        func,
        arr_size=arr_size, unpack_params=unpack_params,
        args=args, kwargs=kwargs,
        log_progress_element=log_progress_element, log_progress_percentage=log_progress_percentage
    )

    start_time = time.perf_counter()
    # start_datetime = datetime.datetime.now().astimezone().isoformat()
    try:
        with get_process_pool(max_workers=max_workers, single_thread=single_thread, global_pool=global_pool) as ex:
            if log_start_finish:
                logger.info(
                    "Starting parallel processing. Executor: %s, workers: %s, tasks: %s",
                    ex, n_workers, n_tasks
                )

            # -----
            # Submit parallel execution
            # -----
            with np.nditer(
                    [params, None],
                    flags=flags,
                    op_flags=[["readonly"], ["readwrite", "allocate"]],
                    op_axes=op_axes,
                    op_dtypes=[params.dtype, object],
                    order="C") as it:
                for ind, (param, fut) in enumerate(it):
                    multi_index = None if multiple_params else it.multi_index
                    fut[...] = ex.submit(runner.run, param, index=ind, multi_index=multi_index)
                futs = it.operands[1]

            # -----
            # Collect results
            # -----

            # No output
            # if output_dtypes is None:
            #     return None

            # Array output
            if return_arr_shape is not None:
                if output_dtypes is None or not len(output_dtypes):
                    raise ValueError("Please give the output dtype.")
                elif len(output_dtypes) > 1:
                    raise ValueError("Array output is currently supported for only one array.")
                output_arr = np.empty((*futs.shape, *return_arr_shape), dtype=output_dtypes[0])
                # axes = list(range(futs.ndim)) # + list((-1, ) * len(return_arr_shape))
                # op_axes = [axes, axes]
                with np.nditer(
                        futs,
                        flags=["refs_ok", "c_index", "multi_index"],
                        # op_flags=[["readonly"], ["writeonly"]],
                        # op_axes=op_axes,
                        order="C") as it:
                    for fut in it:
                        output_arr[*it.multi_index, :] = fut.item().result()
                if log_start_finish:
                    log_parallel_ready(executor=ex, n_workers=n_workers, n_tasks=n_tasks, start_time=start_time)
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
                        flags=["refs_ok", "c_index", "multi_index"],
                        # op_flags=[["readonly"], ["writeonly"]],
                        order="C") as it:
                    for fut, res in it:
                        res[...] = fut.item().result()
                    if log_start_finish:
                        log_parallel_ready(executor=ex, n_workers=n_workers, n_tasks=n_tasks, start_time=start_time)
                    return it.operands[1]

            # Multiple outputs
            op_flags2 = [["readonly"], *[["writeonly"]] * len(output_dtypes)]
            output_arrs = tuple(np.empty(futs.shape, dtype=dtype) for dtype in output_dtypes)
            with np.nditer(
                    [futs, *output_arrs],
                    flags=["refs_ok", "c_index", "multi_index"],
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
            if log_start_finish:
                log_parallel_ready(executor=ex, n_workers=n_workers, n_tasks=n_tasks, start_time=start_time)
            return output_arrs
    except BrokenProcessPool as err:
        msg = parallel_debug_message(
            info="Parallel execution failed due to a system error. ",
            err=err,
            max_workers=max_workers,
            single_thread=single_thread,
            start_time=start_time,
            kwargs={
                # Internal variables
                "flags": flags,
                "arr_size": arr_size,
                "op_axes": op_axes,
                # Arguments
                "func": func,
                "params.shape": params.shape,
                # "params": params,
                "multiple_params": multiple_params,
                "unpack_params": unpack_params,
                "output_dtypes": output_dtypes,
                "return_arr_shape": return_arr_shape,
                "args": args,
                "kwargs": kwargs
            }
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
