r"""Fluid speed $\tilde{v}_+$ ahead of the wall in the wall frame

.. plot:: fig/vp_vm_plane.py

Please also see :barni_2026:`\ ` fig. 1 and :barni_2024:`\ ` fig. 3 and 5.
"""

import numba
from numba.extending import overload
import numpy as np

from pttools.bubble.const import CS0
from pttools.bubble.solution_type import SolutionType
from pttools.speedup import NUMBA_ENABLE_CACHE
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr


@numba.njit
def max_speed_deflag(alpha_p: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Maximum speed for a deflagration: speed where wall and shock are coincident.
    May be greater than 1, meaning that hybrids exist for all wall speeds above cs.
    $\alpha_+ < \frac{1}{3}$, but $\alpha_n$ unbounded above.

    :param alpha_p: $\alpha_+$
    """
    return 1 / (3 * v_plus(CS0, alpha_p, SolutionType.SUB_DEF))


def _v_plus_scalar(
        vm: th.FloatOrArr,
        ap: float,
        sol_type: SolutionType,
        debug: bool = True,
        parallel: bool = True) -> th.FloatOrArrNumba:
    x = vm + 1. / (3 * vm)
    # Finding the SolutionType automatically does not work in the general case
    # if sol_type is None:
    #     b = 1. if vm > 1/np.sqrt(3) else -1.
    # else:
    b = 1. if sol_type == SolutionType.DETON.value else -1.
    # Fluid must flow through the wall from the outside to the inside of the bubble.
    if b == -1 and ap > 1/3:
        # Todo: better error handling and logging
        # if debug:
        #     with numba.objmode:
        #         logger.error("v_plus would be negative for a deflagration with ap > 1/3, got ap=%s", ap)
        return np.nan

    return (0.5 / (1 + ap)) * (x + b * np.sqrt(x ** 2 + 4. * ap ** 2 + (8. / 3.) * ap - (4. / 3.)))
    # if vp < 0:
    #     with numba.objmode:
    #         logger.error(
    #             f"Cannot compute v_plus, got negative result: {vp}. "
    #             "THIS SHOULD NOT HAPPEN. Earlier checks should have caught this."
    #         )
    #     return np.nan
    # return vp

    # Handling of complex return values for scalars
    # if np.imag(ret):
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_plus. This is deprecated. "
    #             "Check the types of the arguments.")
    #     return np.nan
    # return ret


_v_plus_scalar_numba = numba.njit(_v_plus_scalar)


def _v_plus_arr(vm: th.FloatOrArr, ap: float, sol_type: SolutionType, debug: bool = True) -> th.FloatOrArrNumba:
    ret = np.empty_like(vm)
    # pylint: disable=not-an-iterable
    for i in numba.prange(vm.size):
        ret[i] = _v_plus_scalar_numba(vm[i], ap, sol_type, debug)
    return ret

    # complex_inds = np.where(np.imag(ret))
    # if np.any(complex_inds):
    #     ret[np.where(np.imag(ret))] = np.nan
    #     with numba.objmode:
    #         logger.warning(
    #             "Complex numbers detected in v_plus. This is deprecated. "
    #             "Check the types of the arguments.")
    # return np.real(ret)


_v_plus_arr_parallel = numba.njit(parallel=True, nogil=True)(_v_plus_arr)
_v_plus_arr_single = numba.njit(nogil=True)(_v_plus_arr)


def _v_plus_arr_wrapper(
        vm: th.FloatOrArr,
        ap: float,
        sol_type: SolutionType,
        debug: bool = True,
        parallel: bool = True) -> th.FloatOrArrNumba:
    if parallel:
        return _v_plus_arr_parallel(vm=vm, ap=ap, sol_type=sol_type, debug=debug)
    return _v_plus_arr_single(vm=vm, ap=ap, sol_type=sol_type, debug=debug)


def v_plus(
        vm: th.FloatOrArr,
        ap: float,
        sol_type: SolutionType,
        debug: bool = True,
        parallel: bool = True) -> th.FloatOrArrNumba:
    r"""
    Fluid speed $\tilde{v}_+$ ahead of the wall in the wall frame
    $$\tilde{v}_+ = \frac{1}{2(1 + \alpha_+)}
    \left[
    \left( \frac{1}{3 \tilde{v}_-} + \tilde{v}_- \right)
    \pm
    \sqrt{ \left( \frac{1}{3\tilde{v}_-} - \tilde{v}_- \right)^2 + 4\alpha_+^2 + \frac{8}{3} \alpha_+}
    \right]$$
    :gw_pt_ssm:`\ `, eq. B.6,
    :notes:`\ `, eq. 7.27.
    The equations in both sources are equivalent by moving a factor of 2.

    Positive sign is for detonations,
    which corresponds to $\tilde{v}_- > \frac{1}{\sqrt{3}}$ in the bag model.

    :param vm: $\tilde{v}_-$, fluid speed behind the wall
    :param ap: $\alpha_+$, strength parameter at the wall
    :param sol_type: Detonation, Deflagration, Hybrid
    :return: $\tilde{v}_+$, fluid speed ahead of the wall
    """
    # TODO: add support for having both arguments as arrays
    if isinstance(vm, float):
        return _v_plus_scalar(vm, ap, sol_type, debug)
    if isinstance(vm, np.ndarray):
        return _v_plus_arr(vm, ap, sol_type, debug)
    raise TypeError(f"Unknown argument types: vm = {type(vm)}, ap = {type(ap)}")


@overload(v_plus, jit_options={"nopython": True, "nogil": True, "cache": NUMBA_ENABLE_CACHE})
def _v_plus_numba(
        vm: th.FloatOrArr,
        ap: float,
        sol_type: SolutionType,
        debug: bool = True,
        parallel: bool = True) -> th.FloatOrArrNumba:
    if isinstance(vm, numba.types.Float):
        return _v_plus_scalar
    if isinstance(vm, numba.types.Array):
        return _v_plus_arr_wrapper
    raise TypeError(f"Unknown argument types: vm = {type(vm)}, ap = {type(ap)}")


def v_plus_limit[T: FloatOrArr](ap: T, sol_type: SolutionType) -> T:
    r"""Limit for the values that $\tilde{v}_+$ can have.

    TODO this is the Chapman-Jouguet speed, not a separate limit!

    $$\frac{1}{1+\alpha_+} \left( \frac{1}{\sqrt{3}} \pm \sqrt{\alpha_+ ( \alpha_+ + \frac{2}{3})} \right)
    """
    b = 1 if sol_type == SolutionType.DETON.value else -1
    return 1/(1 + ap) * (1/np.sqrt(3) + b * np.sqrt(ap * (ap + 2/3)))


def v_plus_off_limits(vp: float, ap: float, sol_type: SolutionType) -> bool:
    if sol_type == SolutionType.DETON.value:
        return vp < v_plus_limit(ap, sol_type)
    return vp > v_plus_limit(ap, sol_type)
