import concurrent.futures as cf
import logging
# import os
import typing as tp

# from numba import get_num_threads, set_num_threads
import numpy as np

from pttools.speedup.options import MAX_WORKERS_DEFAULT  #, GITHUB_ACTIONS

logger = logging.getLogger(__name__)


class FakeFuture:
    def __init__(self, func: callable, *args, **kwargs):
        self._result = func(*args, **kwargs)

    def result(self):
        return self._result


class LoggingRunner:
    """A handler for logging the execution status of a function that is run in parallel"""
    def __init__(
            self,
            func: callable,
            arr_size: int,
            unpack_params: bool,
            args: tuple = (),
            kwargs: tp.Dict[str, any] = None,
            log_progress_element: int = None,
            log_progress_percentage: float = None):
        self.func = func
        self.arr_size = arr_size
        self.unpack_params = unpack_params
        self.args = args
        self.kwargs = {} if kwargs is None else kwargs
        self.log_progress_element = log_progress_element
        self.log_progress_percentage = log_progress_percentage

    def run(self, param, index: int = None, multi_index: tp.Iterable = None):
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
                    logger.debug("Processed item %s, %d/%d, %.2f %%", multi_index, index, self.arr_size, percentage)

        return ret


def run_parallel(
        func: callable,
        params: np.ndarray,
        max_workers: int = MAX_WORKERS_DEFAULT,
        multiple_params: bool = False,
        unpack_params: bool = False,
        output_dtypes: tp.Union[tp.Tuple[tp.Type, ...], tp.List[tp.Type]] = None,
        log_progress_element: int = None,
        log_progress_percentage: float = None,
        args: tp.Union[list, tuple] = (),
        kwargs: tp.Dict[str, tp.Any] = None,
        single_thread: bool = False) -> tp.Union[np.ndarray, tp.Tuple[np.ndarray, ...]]:
    """Run the given function with multiple parameters in parallel

    :param func: The function to be executed in parallel
    :param params: Array of the function parameters
    :param max_workers: Maximum number of worker processes
    :param multiple_params: Whether the last dimension of the parameter array contains multiple parameters for each function call
    :param unpack_params: Whether the multiple parameters should be unpacked before giving them to the function
    :param output_dtypes: If the function has multiple output values, their types should be given here
    :param log_progress_element: Log progress every n element
    :param log_progress_percentage: Log progress every x %
    :param args: common arguments for the function
    :param kwargs: common kwargs for the function
    :param single_thread: disable parallelism for debugging and profiling
    :return: Numpy arrays for each output value
    """
    if kwargs is None:
        kwargs = {}

    flags = ["refs_ok"]
    if multiple_params:
        flags.append("reduce_ok")
        flags.append("external_loop")
        op_axes = [None, [*list(range(params.ndim-1)), -1]]
        arr_size: int = np.prod(params.shape[:-1])
    else:
        flags.append("c_index")
        flags.append("multi_index")
        op_axes = None
        arr_size: int = np.prod(params.shape)

    runner = LoggingRunner(
        func,
        arr_size=arr_size, unpack_params=unpack_params,
        args=args, kwargs=kwargs,
        log_progress_element=log_progress_element, log_progress_percentage=log_progress_percentage
    )

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
