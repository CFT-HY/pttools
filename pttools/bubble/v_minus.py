r"""Fluid speed $\tilde{v}_-$ behind the wall in the wall frame

.. plot:: fig/vp_vm_plane.py

Please also see :barni_2026:`\ ` fig. 1 and :barni_2024:`\ ` fig. 3 and 5.
"""

import numba
from numba.extending import overload
import numpy as np

from pttools.bubble.solution_type import SolutionType
from pttools.speedup import njit
import pttools.type_hints as th


def _v_minus_scalar(
        vp: th.FloatOrArr,
        ap: float,
        sol_type: SolutionType = SolutionType.DETON,
        strong_branch: bool = False,
        debug: bool = False,
        parallel: bool = True) -> th.FloatOrArr:
    # Fluid must flow through the wall from the outside to the inside of the bubble.
    if vp < 0:
        return np.nan
    # Todo: Make this implementation more readable.
    # This has probably been written like this for numerical stability
    vp2 = vp**2
    z = vp2 + 1/3 - ap * (1. - vp2)
    sqrt_arg = z**2 - (4/3) * vp2

    # Way 2
    # x = (1 + ap)*vp + (1 - 3*ap)/(3*vp)
    # sqrt_arg = x**2 - 4/3

    if debug and sqrt_arg < 0:
        # Todo: better error handling and logging
        # with numba.objmode:
        #     logger.error(
        #         "Cannot compute vm, got imaginary result with: vp=%s, ap=%s in sqrt_arg=%s",
        #         vp, ap, sqrt_arg)
        return np.nan

    # Finding the solution type automatically does not work in the general case
    # if sol_type is None:
    #     b = 1. if vp < 1/np.sqrt(3) else -1
    # else:

    b = 1. if sol_type == SolutionType.DETON.value else -1
    c = -1 if strong_branch else 1
    return (0.5 / vp) * (z + b * c * np.sqrt(sqrt_arg))

    # Way 2
    # return 0.5 * (x + b*np.sqrt(sqrt_arg))

    # Handling of complex return values for scalars
    # if np.imag(ret):
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_minus. This is deprecated. "
    #             "Check the types of the arguments.")
    #     return np.nan
    # return ret


_v_minus_scalar_numba = njit(_v_minus_scalar, nogil=True, cache=True)


def _v_minus_arr(
        vp: th.FloatOrArr,
        ap: float,
        sol_type: SolutionType = SolutionType.DETON,
        strong_branch: bool = False,
        debug: bool = False) -> th.FloatArr:
    ret = np.empty_like(vp)
    # pylint: disable=not-an-iterable
    for i in numba.prange(vp.size):
        ret[i] = _v_minus_scalar_numba(vp[i], ap, sol_type, strong_branch, debug)
    return ret

    # complex_inds = np.where(np.imag(ret))
    # if np.any(complex_inds):
    #     ret[np.where(np.imag(ret))] = np.nan
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_minus. This is deprecated. "
    #             "Check the types of the arguments.")
    # return ret

_v_minus_arr_parallel = njit(parallel=True, nogil=True, cache=True)(_v_minus_arr)
_v_minus_arr_single = njit(nogil=True, cache=True)(_v_minus_arr)


def _v_minus_arr_wrapper(
        vp: th.FloatOrArr,
        ap: float,
        sol_type: SolutionType = SolutionType.DETON,
        strong_branch: bool = False,
        debug: bool = False,
        parallel: bool = True) -> th.FloatArr:
    if parallel:
        return _v_minus_arr_parallel(vp=vp, ap=ap, sol_type=sol_type, strong_branch=strong_branch, debug=debug)
    return _v_minus_arr_single(vp=vp, ap=ap, sol_type=sol_type, strong_branch=strong_branch, debug=debug)


def v_minus(
        vp: th.FloatOrArr,
        ap: float,
        sol_type: SolutionType = SolutionType.DETON,
        strong_branch: bool = False,
        debug: bool = False,
        parallel: bool = True) -> th.FloatOrArr:
    r"""
    Fluid speed $\tilde{v}_-$ behind the wall in the wall frame
    $$\tilde{v}_- = \frac{1}{2} \left[
    \left( (1 + \alpha_+)\tilde{v}_+ + \frac{1 - 3\alpha_+}{3 \tilde{v}_+} \right)
    \pm
    \sqrt{ \left( (1 + \alpha_+)\tilde{v}_+ + \frac{1 - 3\alpha_+}{3 \tilde{v}_+} \right)^2 - \frac{4}{3} }
    \right]$$
    :gw_pt_ssm:`\ `, eq. B.7

    Positive sign is for detonations,
    which corresponds to $\tilde{v}_+ < \frac{1}{\sqrt{3}}$ in the bag model.
    TODO Check that this is actually the case.

    :param vp: $\tilde{v}_+$, fluid speed ahead of the wall
    :param ap: $\alpha_+$, strength parameter at the wall
    :param sol_type: Detonation, Deflagration, Hybrid (assumed detonation if not given)
    :return: $\tilde{v}_-$, fluid speed behind the wall
    """
    # TODO: add support for having both arguments as arrays
    if isinstance(vp, float):
        return _v_minus_scalar(vp, ap, sol_type, strong_branch, debug)
    if isinstance(vp, np.ndarray):
        return _v_minus_arr(vp, ap, sol_type, strong_branch, debug)
    raise TypeError(f"Unknown argument types: vp = {type(vp)}, ap = {type(ap)}")


# The Numba caching of the overload implementations is disabled, as their cache files collide
# with those of the njit-compiled versions of the same functions, which results in segmentation faults.
@overload(v_minus, jit_options={"nopython": True, "nogil": True})
def _v_minus_numba(
        vp: th.FloatOrArr,
        ap: float,
        sol_type: SolutionType = SolutionType.DETON,
        strong_branch: bool = False,
        debug: bool = False,
        parallel: bool = True) -> th.NumbaFunc:
    if isinstance(vp, numba.types.Float):
        return _v_minus_scalar
    if isinstance(vp, numba.types.Array):
        return _v_minus_arr_wrapper
    raise TypeError(f"Unknown argument types: vp = {type(vp)}, ap = {type(ap)}")
