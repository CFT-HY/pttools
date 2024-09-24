import concurrent.futures as cf
import logging
import time
import typing as tp

import numpy as np

from pttools.bubble.bubble import Bubble
from pttools.bubble import fluid_reference
from pttools.bubble.integrate import precompile
from pttools.ssmtools.spectrum import DEFAULT_NUC_TYPE, NucType, Spectrum
from pttools.ssmtools.const import Z_ST_THRESH
from pttools.speedup import options
from pttools.speedup import parallel
if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


def create_bubble(
        params: np.ndarray,
        model: "Model",
        post_func: callable = None,
        post_func_return_multiple: bool = False,
        use_bag_solver: bool = False,
        bubble_kwargs: tp.Dict[str, any] = None,
        allow_bubble_failure: bool = False,
        *args, **kwargs) -> tp.Union[tp.Optional[Bubble], tp.Tuple[tp.Optional[Bubble], tp.Any]]:
    v_wall, alpha_n = params
    # This is a common error case and should be handled here to avoid polluting the logs with exceptions.
    if alpha_n < model.alpha_n_min and bubble_kwargs is not None \
            and ("allow_invalid" not in bubble_kwargs or not bubble_kwargs["allow_invalid"]):
        logger.error("Invalid alpha_n=%s. Minimum for the model: %s", alpha_n, model.alpha_n_min)
        return None, post_func.fail_value
    try:
        if bubble_kwargs is None:
            bubble = Bubble(model, v_wall, alpha_n, solve=False)
        else:
            bubble = Bubble(model, v_wall, alpha_n, solve=False, **bubble_kwargs)
    except Exception as e:
        if allow_bubble_failure:
            logger.exception("Failed to create a bubble:", exc_info=e)
            if post_func is None:
                return None
            if post_func_return_multiple:
                return None, *post_func.fail_value
            return None, post_func.fail_value
        raise e
    bubble.solve(use_bag_solver=use_bag_solver)
    if post_func is not None:
        if post_func_return_multiple:
            return bubble, *post_func(bubble, *args, **kwargs)
        return bubble, post_func(bubble, *args, **kwargs)
    return bubble


def create_spectrum(
        params: np.ndarray,
        model: "Model",
        post_func: callable = None,
        post_func_return_multiple: bool = False,
        use_bag_solver: bool = False,
        bubble_kwargs: tp.Dict[str, any] = None,
        allow_bubble_failure: bool = False,
        z: np.ndarray = None,
        z_st_thresh: float = Z_ST_THRESH,
        nuc_type: NucType = DEFAULT_NUC_TYPE,
        *args, **kwargs):
    bubble = create_bubble(
        params=params,
        model=model,
        use_bag_solver=use_bag_solver,
        bubble_kwargs=bubble_kwargs,
        allow_bubble_failure=allow_bubble_failure
    )
    spectrum = Spectrum(bubble=bubble, z=z, z_st_thresh=z_st_thresh, nuc_type=nuc_type)
    if post_func is not None:
        if post_func_return_multiple:
            return spectrum, *post_func(spectrum, *args, **kwargs)
        return spectrum, post_func(spectrum, *args, **kwargs)
    return spectrum


def create_bubbles(
        model: "Model",
        v_walls: np.ndarray,
        alpha_ns: np.ndarray,
        func: callable = None,
        log_progress_percentage: float = 5,
        max_workers: int = options.MAX_WORKERS_DEFAULT,
        allow_bubble_failure: bool = False,
        kwargs: tp.Dict[str, any] = None,
        bubble_kwargs: tp.Dict[str, any] = None,
        bubble_func: callable = create_bubble) -> tp.Union[np.ndarray, tp.Tuple[np.ndarray, np.ndarray]]:
    start_time = time.perf_counter()
    post_func_return_multiple = False
    if func is None:
        output_dtypes = None
    else:
        if not hasattr(func, "return_type"):
            raise ValueError("The function should have a return_type attribute for output array initialization")

        if isinstance(func.return_type, tuple):
            output_dtypes = (object, *func.return_type)
            post_func_return_multiple = True
        else:
            output_dtypes = (object, func.return_type)

    kwargs2 = {
        "model": model,
        "post_func": func,
        "post_func_return_multiple": post_func_return_multiple,
        "bubble_kwargs": bubble_kwargs,
        "allow_bubble_failure": allow_bubble_failure
    }
    if kwargs is not None:
        kwargs2.update(kwargs)

    params = np.empty((alpha_ns.size, v_walls.size, 2))
    for i_alpha_n, alpha_n in enumerate(alpha_ns):
        for i_v_wall, v_wall in enumerate(v_walls):
            params[i_alpha_n, i_v_wall, 0] = v_wall
            params[i_alpha_n, i_v_wall, 1] = alpha_n

    # Pre-do shared steps so that they don't have to be done for each process
    fluid_reference.ref()
    model.df_dtau_ptr()
    precompile()

    # Run the parallel processing
    ret = parallel.run_parallel(
        bubble_func, params,
        multiple_params=True,
        output_dtypes=output_dtypes,
        max_workers=max_workers,
        log_progress_percentage=log_progress_percentage,
        kwargs=kwargs2
    )
    bubble_count = alpha_ns.size * v_walls.size
    elapsed = time.perf_counter() - start_time
    elapsed_per_bubble = elapsed / bubble_count
    logger.debug("Creating %s bubbles took %s s in total, %s s per bubble", bubble_count, elapsed, elapsed_per_bubble)
    return ret


def create_spectra(
        model: "Model",
        v_walls: np.ndarray,
        alpha_ns: np.ndarray,
        func: callable = None,
        log_progress_percentage: float = 5,
        max_workers: int = options.MAX_WORKERS_DEFAULT,
        allow_bubble_failure: bool = False,
        kwargs: tp.Dict[str, any] = None,
        bubble_kwargs: tp.Dict[str, any] = None):
    return create_bubbles(
        model=model,
        v_walls=v_walls,
        alpha_ns=alpha_ns,
        func=func,
        log_progress_percentage=log_progress_percentage,
        max_workers=max_workers,
        allow_bubble_failure=allow_bubble_failure,
        kwargs=kwargs,
        bubble_kwargs=bubble_kwargs,
        bubble_func=create_spectrum
    )


def solve_bubbles(bubbles: np.ndarray, max_workers: int = options.MAX_WORKERS_DEFAULT) -> None:
    futs: tp.List[cf.Future] = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        for bubble in np.nditer(bubbles):
            futs.append(ex.submit(bubble.solve))
        cf.wait(futs)
