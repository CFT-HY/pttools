import concurrent.futures as cf
import logging
import typing as tp

import numpy as np

from pttools.bubble.bubble import Bubble
from pttools.bubble import fluid_reference
from pttools.speedup import options
from pttools.speedup import parallel
if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


def create_bubble(params: np.ndarray, model: "Model", post_func: callable = None, *args, **kwargs):
    v_wall, alpha_n = params
    bubble = Bubble(model, v_wall, alpha_n)
    bubble.solve()
    if post_func is not None:
        return bubble, post_func(bubble, *args, **kwargs)
    return bubble


def create_bubbles(
        model: "Model",
        v_walls: np.ndarray,
        alpha_ns: np.ndarray,
        func: callable = None,
        output_dtypes: tp.Iterable = None,
        max_workers: int = options.MAX_WORKERS_DEFAULT,
        kwargs: tp.Dict[str, any] = None) -> tp.Union[np.ndarray, tp.Tuple[np.ndarray, np.ndarray]]:
    kwargs2 = {
        "model": model,
        "post_func": func
    }
    if kwargs is not None:
        kwargs2.update(kwargs)

    params = np.empty((v_walls.size, alpha_ns.size, 2))
    for i_v_wall, v_wall in enumerate(v_walls):
        for i_alpha_n, alpha_n in enumerate(alpha_ns):
            params[i_v_wall, i_alpha_n, 0] = v_wall
            params[i_v_wall, i_alpha_n, 1] = alpha_n

    fluid_reference.ref()
    return parallel.run_parallel(
        create_bubble, params,
        multiple_params=True,
        output_dtypes=output_dtypes,
        max_workers=max_workers,
        kwargs=kwargs2
    )


def solve_bubbles(bubbles: np.ndarray, max_workers: int = options.MAX_WORKERS_DEFAULT) -> None:
    futs: tp.List[cf.Future] = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        for bubble in np.nditer(bubbles):
            futs.append(ex.submit(bubble.solve))
        cf.wait(futs)
