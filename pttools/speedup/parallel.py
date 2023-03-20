import concurrent.futures as cf
import logging
import typing as tp

import numpy as np

from pttools.speedup.options import MAX_WORKERS_DEFAULT

logger = logging.getLogger(__name__)


def log_progress(it: np.nditer, log_progress_element: int, log_progress_percentage: float):
    ind = it.index
    arr_size = it.operands[0].size
    percentage = ind / arr_size * 100
    percentage_prev = (ind - 1) / arr_size * 100

    if (log_progress_element is not None and ind % log_progress_element == 0) \
            or (log_progress_percentage is not None and
                np.floor(percentage / log_progress_percentage) != np.floor(percentage_prev / log_progress_percentage)):
        logger.debug("Processing item %s, %s %%", it.multi_index, percentage)


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
        kwargs: tp.Dict[str, tp.Any] = None) -> tp.Union[np.ndarray, tp.Tuple[np.ndarray, ...]]:
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
    :return: Numpy arrays for each output value
    """
    if kwargs is None:
        kwargs = {}

    flags = ["refs_ok"]
    if multiple_params:
        flags.append("reduce_ok")
        flags.append("external_loop")
        op_axes = [None, [*list(range(params.ndim-1)), -1]]
    else:
        op_axes = None

    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        # Submit parallel execution
        with np.nditer(
                [params, None],
                flags=flags,
                op_flags=[["readonly"], ["readwrite", "allocate"]],
                op_axes=op_axes,
                op_dtypes=[params.dtype, object]) as it:
            if unpack_params:
                for param, fut in it:
                    fut[...] = ex.submit(func, *param, *args, **kwargs)
            else:
                for param, fut in it:
                    fut[...] = ex.submit(func, param, *args, **kwargs)
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
                    log_progress(it, log_progress_element, log_progress_percentage)
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
                log_progress(it, log_progress_element, log_progress_percentage)
                res = elems[0].item().result()
                for arr, val in zip(elems[1:], res):
                    arr[...] = val
        return output_arrs
