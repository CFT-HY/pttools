import concurrent.futures as cf
import typing as tp

import numpy as np

from pttools.speedup.options import MAX_WORKERS_DEFAULT


def run_parallel(
        func: callable,
        params: np.ndarray,
        max_workers: int = MAX_WORKERS_DEFAULT,
        multiple_params: bool = False,
        output_dtypes: list = None,
        *args,
        **kwargs) -> tp.Union[np.ndarray, tp.Tuple[np.ndarray, ...]]:
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
            for obj, fut in it:
                fut[...] = ex.submit(func, obj, *args, **kwargs)
            futs = it.operands[1]

        # Collect results

        # Single output
        if output_dtypes is None:
            with np.nditer([futs, None], flags=("refs_ok",)) as it:
                for fut, res in it:
                    res[...] = fut.item().result()
                return it.operands[1]

        # Multiple outputs
        op_flags2 = [["readonly"], *[["writeonly"]] * len(output_dtypes)]
        output_arrs = tuple(np.empty(futs.shape, dtype=dtype) for dtype in output_dtypes)
        with np.nditer(
                [futs, *output_arrs],
                flags=("refs_ok",),
                op_flags=op_flags2) as it:
            for elems in it:
                res = elems[0].item().result()
                for arr, val in zip(elems[1:], res):
                    arr[...] = val
            return output_arrs
