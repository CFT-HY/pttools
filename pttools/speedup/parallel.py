import concurrent.futures as cf

import numpy as np

from pttools.speedup.options import MAX_WORKERS_DEFAULT


def run_parallel(
        func: callable,
        params: np.ndarray,
        max_workers: int = MAX_WORKERS_DEFAULT,
        multiple_params: bool = False,
        *args,
        **kwargs) -> np.ndarray:
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
            # it.operands[1][0, 0] = None
            for obj, fut in it:
                fut[...] = ex.submit(func, obj, *args, **kwargs)
            futs = it.operands[1]
        # Collect results
        with np.nditer([futs, None], flags=("refs_ok",)) as it:
            for fut, res in it:
                res[...] = fut.item().result()
            return it.operands[1]
