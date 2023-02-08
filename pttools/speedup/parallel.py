import concurrent.futures as cf
import typing as tp

import numpy as np

from pttools.speedup.options import MAX_WORKERS_DEFAULT


def solve_bubbles(bubbles: np.ndarray, max_workers: int = MAX_WORKERS_DEFAULT) -> None:
    futs: tp.List[cf.Future] = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        for bubble in np.nditer(bubbles):
            futs.append(ex.submit(bubble.solve))
        cf.wait(futs)


def run_parallel(
        func: callable,
        params: np.ndarray,
        max_workers: int = MAX_WORKERS_DEFAULT,
        *args,
        **kwargs) -> np.ndarray:
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        # Submit parallel execution
        with np.nditer([params, None], flags=("refs_ok",)) as it:
            for obj, fut in it:
                fut[...] = ex.submit(func, obj, *args, **kwargs)
            futs = it.operands[1]
        # Collect results
        with np.nditer([futs, None], flags=("refs_ok",)) as it:
            for fut, res in it:
                res[...] = fut.item().result()
            return it.operands[1]
