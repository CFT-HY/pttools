"""Utilities for parallel simulation of bubbles."""

import logging
import time
import typing as tp

import numpy as np
from numpy.typing import NDArray

from pttools.bubble import fluid_reference
from pttools.bubble.bubble import Bubble, BubbleArr, BubbleArr2D
from pttools.bubble.integrate import precompile
from pttools.omgw0 import Spectrum, SpectrumArr2D
from pttools.speedup import options
from pttools.speedup.parallel import run_parallel
import pttools.type_hints as th
from pttools.utils.decorators import PostFunc

if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


def _failure_output(post_func: PostFunc | None, post_func_return_multiple: bool) -> tuple[None, ...] | None:
    """Output of :func:`create_bubble` and :func:`create_spectrum` when the object could not be created."""
    if post_func is None:
        return None
    if post_func_return_multiple:
        return None, *post_func.fail_value
    return None, post_func.fail_value


def _post_process[T: (Bubble, Spectrum)](
        obj: T,
        post_func: PostFunc | None,
        post_func_return_multiple: bool,
        args: tuple[tp.Any, ...],
        kwargs: dict[str, tp.Any]) -> T | tuple[T, ...]:
    """Apply the post-processing function to the output of :func:`create_bubble` or :func:`create_spectrum`."""
    if post_func is None:
        return obj
    if post_func_return_multiple:
        return obj, *post_func(obj, *args, **kwargs)
    return obj, post_func(obj, *args, **kwargs)


def create_bubble(
        params: th.FloatArr1D,
        model: "Model",
        post_func: PostFunc | None = None,
        post_func_return_multiple: bool = False,
        use_bag_solver: bool = False,
        bubble_kwargs: dict[str, tp.Any] | None = None,
        allow_bubble_failure: bool = False,
        *args, **kwargs) -> Bubble | tuple[Bubble | None, ...] | None:
    """Create a single bubble and apply post-processing functions to retrieve results from it."""
    v_wall, alpha_n = params
    # This is a common error case and should be handled here to avoid polluting the logs with exceptions.
    if alpha_n < model.alpha_n_min and bubble_kwargs is not None \
            and ("allow_invalid" not in bubble_kwargs or not bubble_kwargs["allow_invalid"]):
        logger.error("Invalid alpha_n=%s. Minimum for the model: %s", alpha_n, model.alpha_n_min)
        return _failure_output(post_func, post_func_return_multiple)
    try:
        if bubble_kwargs is None:
            bubble = Bubble(model, v_wall, alpha_n, solve=False)
        else:
            bubble = Bubble(model, v_wall, alpha_n, solve=False, **bubble_kwargs)
    except Exception as e:
        if allow_bubble_failure:
            logger.exception("Failed to create a bubble:", exc_info=e)
            return _failure_output(post_func, post_func_return_multiple)
        raise e
    bubble.solve(use_bag_solver=use_bag_solver)
    return _post_process(bubble, post_func, post_func_return_multiple, args, kwargs)


def create_spectrum(
        params: th.FloatArr1D,
        model: "Model",
        post_func: PostFunc | None = None,
        post_func_return_multiple: bool = False,
        use_bag_solver: bool = False,
        bubble_kwargs: dict[str, tp.Any]| None  = None,
        spectrum_kwargs: dict[str, tp.Any] | None = None,
        allow_bubble_failure: bool = False,
        *args, **kwargs) -> Spectrum | tuple[Spectrum | None, ...] | None:
    """Create a single spectrum and apply post-processing functions to retrieve results from it."""
    bubble = create_bubble(
        params=params,
        model=model,
        use_bag_solver=use_bag_solver,
        bubble_kwargs=bubble_kwargs,
        allow_bubble_failure=allow_bubble_failure
    )
    # Without post_func, create_bubble() returns None if it fails to create the bubble and allow_bubble_failure is set.
    if not isinstance(bubble, Bubble):
        return _failure_output(post_func, post_func_return_multiple)
    if spectrum_kwargs is None:
        spectrum = Spectrum(bubble=bubble, parallel=False)
    else:
        spectrum = Spectrum(bubble=bubble, parallel=False, **spectrum_kwargs)

    return _post_process(spectrum, post_func, post_func_return_multiple, args, kwargs)


@tp.overload
def create_bubbles(
        model: "Model",
        v_walls: th.FloatArr1D,
        alpha_ns: th.FloatArr1D,
        func: None = None,
        log_progress_percentage: float = ...,
        max_workers: int = ...,
        single_thread: bool = ...,
        allow_bubble_failure: bool = ...,
        kwargs: dict[str, tp.Any] | None = ...,
        bubble_kwargs: dict[str, tp.Any] | None = ...,
        bubble_func: tp.Callable = ...) -> BubbleArr2D: ...


@tp.overload
def create_bubbles(
        model: "Model",
        v_walls: th.FloatArr1D,
        alpha_ns: th.FloatArr1D,
        func: tp.Callable,
        log_progress_percentage: float = ...,
        max_workers: int = ...,
        single_thread: bool = ...,
        allow_bubble_failure: bool = ...,
        kwargs: dict[str, tp.Any] | None = ...,
        bubble_kwargs: dict[str, tp.Any] | None = ...,
        bubble_func: tp.Callable = ...) -> tuple[NDArray, ...]: ...


@tp.overload
def create_bubbles(
        model: "Model",
        v_walls: th.FloatArr1D,
        alpha_ns: th.FloatArr1D,
        func: tp.Callable | None = ...,
        log_progress_percentage: float = ...,
        max_workers: int = ...,
        single_thread: bool = ...,
        allow_bubble_failure: bool = ...,
        kwargs: dict[str, tp.Any] | None = ...,
        bubble_kwargs: dict[str, tp.Any] | None = ...,
        bubble_func: tp.Callable = ...) -> BubbleArr2D | tuple[NDArray, ...]: ...


def create_bubbles(
        model: "Model",
        v_walls: th.FloatArr1D,
        alpha_ns: th.FloatArr1D,
        func: tp.Callable | None = None,
        log_progress_percentage: float = 10,
        max_workers: int = options.MAX_WORKERS_DEFAULT,
        single_thread: bool = False,
        allow_bubble_failure: bool = False,
        kwargs: dict[str, tp.Any] | None = None,
        bubble_kwargs: dict[str, tp.Any] | None = None,
        bubble_func: tp.Callable = create_bubble) -> BubbleArr2D | tuple[NDArray, ...]:
    """Create multiple bubbles in parallel."""
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

    # Pre-do shared steps so that they don't have to be done for each process
    fluid_reference.ref()
    # model.df_dtau_ptr()  # This should no longer be necessary
    precompile()

    # Run the parallel processing
    ret = run_parallel(
        bubble_func, params=v_wall_alpha_n_grid(v_walls=v_walls, alpha_ns=alpha_ns),
        multiple_params=True,
        output_dtypes=output_dtypes,
        max_workers=max_workers,
        single_thread=single_thread,
        log_progress_percentage=log_progress_percentage,
        kwargs=kwargs2
    )
    if ret is None:
        raise RuntimeError("The parallel processing returned no output.")
    bubble_count = alpha_ns.size * v_walls.size
    elapsed = time.perf_counter() - start_time
    elapsed_per_bubble = elapsed / bubble_count
    logger.debug(
        "Creating %s bubbles took %.3f s in total, %.3f s per bubble",
        bubble_count, elapsed, elapsed_per_bubble
    )
    return ret


def create_spectra(
        model: "Model",
        v_walls: th.FloatArr1D,
        alpha_ns: th.FloatArr1D,
        func: tp.Callable | None = None,
        log_progress_percentage: float = 5,
        max_workers: int = options.MAX_WORKERS_DEFAULT,
        single_thread: bool = False,
        allow_bubble_failure: bool = False,
        kwargs: dict[str, tp.Any] | None = None,
        bubble_kwargs: dict[str, tp.Any] | None = None,
        spectrum_kwargs: dict[str, tp.Any] | None = None) -> SpectrumArr2D | tuple[NDArray, ...]:
    """Create multiple spectra in parallel."""
    if kwargs is None:
        kwargs2 = {"spectrum_kwargs": spectrum_kwargs}
    else:
        kwargs2 = kwargs.copy()
        kwargs2["spectrum_kwargs"] = spectrum_kwargs
    # The output array contains spectra instead of bubbles, since the bubble_func is create_spectrum().
    return create_bubbles(  # pyrefly: ignore[bad-return]
        model=model,
        v_walls=v_walls,
        alpha_ns=alpha_ns,
        func=func,
        log_progress_percentage=log_progress_percentage,
        max_workers=max_workers,
        single_thread=single_thread,
        allow_bubble_failure=allow_bubble_failure,
        kwargs=kwargs2,
        bubble_kwargs=bubble_kwargs,
        bubble_func=create_spectrum
    )


def solve_bubble(bubble: Bubble) -> None:
    """Solve a single existing bubble."""
    bubble.solve()


def solve_bubbles(bubbles: BubbleArr, max_workers: int = options.MAX_WORKERS_DEFAULT) -> None:
    """Solve multiple existing bubbles in parallel."""
    run_parallel(solve_bubble, params=bubbles, max_workers=max_workers)


def v_wall_alpha_n_grid(v_walls: th.FloatArr1D, alpha_ns: th.FloatArr1D) -> th.FloatArr3D:
    params = np.empty((alpha_ns.size, v_walls.size, 2))
    for i_alpha_n, alpha_n in enumerate(alpha_ns):
        for i_v_wall, v_wall in enumerate(v_walls):
            params[i_alpha_n, i_v_wall, 0] = v_wall
            params[i_alpha_n, i_v_wall, 1] = alpha_n
    return params
