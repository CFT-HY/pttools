import concurrent.futures as cf
import logging
import typing as tp

import numpy as np

from pttools.speedup.options import MAX_WORKERS_DEFAULT

logger = logging.getLogger(__name__)


def run_parallel(
        func: callable,
        params: np.ndarray,
        max_workers: int = MAX_WORKERS_DEFAULT,
        multiple_params: bool = False,
        unpack_params: bool = False,
        output_dtypes: tp.Union[tp.Tuple[tp.Type, ...], tp.List[tp.Type]] = None,
        log_progress: bool = False,
        *args,
        **kwargs) -> tp.Union[np.ndarray, tp.Tuple[np.ndarray, ...]]:
    """Run the given function with multiple parameters in parallel

    :param func: The function to be executed in parallel
    :param params: Array of the function parameters
    :param max_workers: Maximum number of worker processes
    :param multiple_params: Whether the last dimension of the parameter array contains multiple parameters for each function call
    :param unpack_params: Whether the multiple parameters should be unpacked before giving them to the function
    :param output_dtypes: If the function has multiple output values, their types should be given here
    :param log_progress: Whether to output progress to logging
    :return: Numpy arrays for each output value
    """
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
        if output_dtypes is None:
            with np.nditer(
                    [futs, None],
                    flags=("refs_ok", "multi_index")) as it:
                for fut, res in it:
                    if log_progress:
                        logger.debug("Processing item %s", it.multi_index)
                    res[...] = fut.item().result()
                return it.operands[1]

        # Multiple outputs
        op_flags2 = [["readonly"], *[["writeonly"]] * len(output_dtypes)]
        output_arrs = tuple(np.empty(futs.shape, dtype=dtype) for dtype in output_dtypes)
        with np.nditer(
                [futs, *output_arrs],
                flags=("refs_ok", "multi_index"),
                op_flags=op_flags2) as it:
            for elems in it:
                if log_progress:
                    logger.debug("Processing item %s", it.multi_index)
                res = elems[0].item().result()
                for arr, val in zip(elems[1:], res):
                    arr[...] = val
        return output_arrs
