"""Thermodynamic quantities for the Bag Model"""

import logging

import typing as tp

import numba
from numba.extending import overload
import numpy as np

from pttools.bubble import bag
from pttools.bubble.cs2_bag import CS2_BAG_SCALAR_PTR, cs2_bag_scalar
from pttools.bubble.phase import Phase, get_phase
from pttools.bubble import check
from pttools.bubble import const
from pttools.bubble import fluid_bag
from pttools.bubble.integrate import DEFAULT_FLUID_INTEGRATE_METHOD, DF_DTAU_PTR_BAG, FluidIntegrateMethod
from pttools.bubble.thermo import kinetic_energy_density, mean_enthalpy_change, ubarf2
from pttools.bubble.solution_type_bag import SolutionType, identify_solution_type_bag
import pttools.type_hints as th
from pttools.speedup.differential import DifferentialPointer
from pttools.speedup.jit import njit, njit_parallel_pair
from pttools.type_hints import FloatOrArr, FloatOrArr1D

type Integrand = \
    tp.Callable[
        [th.FloatArr1D, th.FloatArr1D, th.FloatArr1D],
        th.FloatArr1D
    ] | \
    tp.Callable[
        [float, float, float],
        float
    ]

logger = logging.getLogger(__name__)


@njit
def de_from_w_bag(
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        alpha_n: float,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_fun: th.CS2Fun) -> th.FloatArr1D:
    r"""
    Calculates energy density difference ``de = e - e[-1]`` from enthalpy, assuming
    bag equation of state.
    Can get ``alpha_n = find_alpha_n_from_w_xi(w,xi,v_wall,alpha_p)``

    :param w: $w$
    :param xi: $\xi$
    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param df_dtau_ptr: pointer to the differential equation function
    :param ode_method: differential equation solver to be used
    :param cs2_fun: $c_s^2$ function
    :return: energy density difference de
    """
    check.check_physical_params(
        (v_wall, alpha_n), df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, cs2_fun=cs2_fun)
    e_from_w = bag.e_bag(w=w, phase=get_phase(xi, v_wall), theta_s=0.75 * w[-1] * alpha_n)

    return e_from_w - e_from_w[-1]


@njit
def de_from_w_new_bag(
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        alpha_n: float,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_fun: th.CS2Fun) -> th.FloatArr1D:
    r"""
    For exploring new methods of calculating energy density difference
    from velocity and enthalpy, assuming bag equation of state.

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param df_dtau_ptr: pointer to the differential equation function
    :param ode_method: differential equation solver to be used
    :param cs2_fun: $c_s^2$ function
    :return: energy density difference de
    """
    check.check_physical_params(
        (v_wall, alpha_n), df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, cs2_fun=cs2_fun)
    e_from_w = bag.e_bag(w=w, phase=get_phase(xi, v_wall), theta_s=0.75 * w[-1] * alpha_n)

    de = e_from_w - e_from_w[-1]

    # Try adjusting by a factor - currently doesn't do anything
    # de *= 1.0

    return de


def get_kappa_bag[T: FloatOrArr](
        v_wall: T,
        alpha_n: float,
        n_xi: int = const.DEFAULT_N_XI,
        verbosity: int = 0) -> T:
    r"""
    Efficiency factor $\kappa$ from $v_\text{wall}$ and $\alpha_n$.

    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param n_xi: number of $\xi$ points
    :param verbosity: logging verbosity
    :return: efficiency factor $\kappa$
    """
    # NB was called get_kappa_arr
    it = np.nditer([v_wall, None])
    for vw, kappa in it:
        # This is necessary for Numba
        vw = vw.item()
        sol_type = identify_solution_type_bag(
            vw, alpha_n, df_dtau_ptr=DF_DTAU_PTR_BAG, ode_method=DEFAULT_FLUID_INTEGRATE_METHOD,
            cs2_fun=cs2_bag_scalar)

        if not sol_type == SolutionType.ERROR:
            # Now ready to solve for fluid profile
            v, w, xi = fluid_bag.sound_shell_bag(
                vw, alpha_n, cs2_fun_ptr=CS2_BAG_SCALAR_PTR, df_dtau_ptr=DF_DTAU_PTR_BAG,
                ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_fun=cs2_bag_scalar, n_xi=n_xi)

            kappa[...] = ubarf2(v, w, xi, vw, w_bar=w[-1]) / (0.75 * alpha_n)
        else:
            kappa[...] = np.nan
        if verbosity > 0:
            logger.debug(
                "%8.6f %8.6f %f",
                vw, alpha_n, kappa
            )

    kappa_out: T
    if isinstance(v_wall, np.ndarray):
        # typing.cast() is not used below, since Numba cannot compile it.
        kappa_out = it.operands[1]  # type: ignore[assignment]
    else:
        kappa_out = type(v_wall)(it.operands[1])

    return kappa_out


def get_kappa_de_bag[T: FloatOrArr](
        v_wall: T,
        alpha_n: float,
        n_xi: int = const.DEFAULT_N_XI,
        verbosity: int = 0) -> tuple[T, T]:
    r"""
    Calculates efficiency factor $\kappa$ and fractional change in energy
    from $v_\text{wall}$ and $\alpha_n$. $v_\text{wall}$ can be an array.
    Sum should be 0 (bag model).

    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param n_xi: number of $\xi$ points
    :param verbosity: logging verbosity
    :return: $\kappa, de$
    """
    it = np.nditer([v_wall, None, None])
    for vw, kappa, de in it:
        vw = vw.item()
        sol_type = identify_solution_type_bag(
            vw, alpha_n, df_dtau_ptr=DF_DTAU_PTR_BAG, ode_method=DEFAULT_FLUID_INTEGRATE_METHOD,
            cs2_fun=cs2_bag_scalar)

        if not sol_type == SolutionType.ERROR:
            # Now ready to solve for fluid profile
            v, w, xi = fluid_bag.sound_shell_bag(
                vw, alpha_n, cs2_fun_ptr=CS2_BAG_SCALAR_PTR, df_dtau_ptr=DF_DTAU_PTR_BAG,
                ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_fun=cs2_bag_scalar, n_xi=n_xi)
            # Esp+ epsilon is alpha_n * 0.75*w_n
            kappa[...] = ubarf2(v, w, xi, vw, w_bar=w[-1]) / (0.75 * alpha_n)
            de[...] = mean_energy_change_bag(v, w, xi, vw, alpha_n)
        else:
            kappa[...] = np.nan
            de[...] = np.nan
        if verbosity > 0:
            logger.debug(
                "%8.6f %8.6f %f %f",
                vw, alpha_n, kappa, de
            )

    kappa_out: T
    de_out: T
    if isinstance(v_wall, np.ndarray):
        kappa_out = it.operands[1]  # type: ignore[assignment]
        de_out = it.operands[2]  # type: ignore[assignment]
    else:
        kappa_out = type(v_wall)(it.operands[1])
        de_out = type(v_wall)(it.operands[2])

    return kappa_out, de_out


def get_kappa_dq_bag[T: FloatOrArr](
        v_wall: T,
        alpha_n: float,
        n_xi: int = const.DEFAULT_N_XI,
        verbosity: int = 0) -> tuple[T, T]:
    r"""
    Calculates efficiency factor $\kappa$ and fractional change in thermal energy
    from $v_\text{wall}$ and $\alpha_n$.
    $v_\text{wall}$ can be an array.
    Sum should be 1.
    Thermal energy is defined as $q = \frac{3}{4} \text{enthalpy}$.

    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param n_xi: number of $\xi$ points
    :param verbosity: logging verbosity
    :return: $\kappa$, dq
    """
    it = np.nditer([v_wall, None, None])
    for vw, kappa, dq in it:
        vw = vw.item()
        sol_type = identify_solution_type_bag(
            vw, alpha_n, df_dtau_ptr=DF_DTAU_PTR_BAG, ode_method=DEFAULT_FLUID_INTEGRATE_METHOD,
            cs2_fun=cs2_bag_scalar)

        if not sol_type == SolutionType.ERROR:
            # Now ready to solve for fluid profile
            v, w, xi = fluid_bag.sound_shell_bag(
                vw, alpha_n, cs2_fun_ptr=CS2_BAG_SCALAR_PTR, df_dtau_ptr=DF_DTAU_PTR_BAG,
                ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_fun=cs2_bag_scalar, n_xi=n_xi)
            # Esp+ epsilon is alpha_n * 0.75*w_n
            kappa[...] = ubarf2(v, w, xi, vw, w_bar=w[-1]) / (0.75 * alpha_n)
            dq[...] = 0.75 * mean_enthalpy_change(v, w, xi, vw) / (0.75 * alpha_n * w[-1])
        else:
            kappa[...] = np.nan
            dq[...] = np.nan
        if verbosity > 0:
            logger.debug(
                "%8.6f %8.6f %f %f",
                vw, alpha_n, kappa, dq
            )

    kappa_out: T
    dq_out: T
    if isinstance(v_wall, np.ndarray):
        kappa_out = it.operands[1]  # type: ignore[assignment]
        dq_out = it.operands[2]  # type: ignore[assignment]
    else:
        kappa_out = type(v_wall)(it.operands[1])
        dq_out = type(v_wall)(it.operands[2])

    return kappa_out, dq_out


def get_ke_de_frac_bag[T: FloatOrArr](
        v_wall: T,
        alpha_n: float,
        n_xi: int = const.DEFAULT_N_XI,
        verbosity: int = 0) -> tuple[T, T]:
    r"""
    Kinetic energy fraction and fractional change in energy
    from wall velocity array. Sum should be 0. Assumes bag model.

    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param n_xi: number of $\xi$ points
    :param verbosity: logging verbosity
    :return: kinetic energy fraction, fractional change in energy
    """
    it = np.nditer([v_wall, None, None])
    for vw, ke, de in it:
        vw = vw.item()
        sol_type = identify_solution_type_bag(
            vw, alpha_n, df_dtau_ptr=DF_DTAU_PTR_BAG, ode_method=DEFAULT_FLUID_INTEGRATE_METHOD,
            cs2_fun=cs2_bag_scalar)

        if not sol_type == SolutionType.ERROR:
            # Now ready to solve for fluid profile
            v, w, xi = fluid_bag.sound_shell_bag(
                vw, alpha_n, cs2_fun_ptr=CS2_BAG_SCALAR_PTR, df_dtau_ptr=DF_DTAU_PTR_BAG,
                ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_fun=cs2_bag_scalar, n_xi=n_xi)
            # Esp+ epsilon is alpha_n * 0.75*w_n
            ke[...] = ubarf2(v, w, xi, vw, w_bar=w[-1]) / (0.75 * (1 + alpha_n))
            de[...] = mean_energy_change_bag(v, w, xi, vw, alpha_n) / (0.75 * w[-1] * (1 + alpha_n))
        else:
            ke[...] = np.nan
            de[...] = np.nan
        if verbosity > 0:
            logger.debug(
                "%8.6f %8.6f %f %f",
                vw, alpha_n, ke, de
            )

    ke_out: T
    de_out: T
    if isinstance(v_wall, np.ndarray):
        ke_out = it.operands[1]  # type: ignore[assignment]
        de_out = it.operands[2]  # type: ignore[assignment]
    else:
        ke_out = type(v_wall)(it.operands[1])
        de_out = type(v_wall)(it.operands[2])

    return ke_out, de_out


def get_ke_frac_bag[T: FloatOrArr](v_wall: T, alpha_n: float, n_xi: int = const.DEFAULT_N_XI) -> T:
    r"""
    Determine kinetic energy fraction (of total energy).
    Bag equation of state only so far, as it takes
    $e_n = \frac{3}{4} w_n (1 + \alpha_n)$.
    This assumes zero trace anomaly in broken phase.

    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param n_xi: number of $\xi$ points
    :return: kinetic energy fraction
    """
    ubarf2 = get_ubarf2_bag(
        v_wall, alpha_n, cs2_fun_ptr=CS2_BAG_SCALAR_PTR, df_dtau_ptr=DF_DTAU_PTR_BAG,
        ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_fun=cs2_bag_scalar, n_xi=n_xi)
    return ubarf2 / (0.75 * (1 + alpha_n))  # type: ignore[return-value]


def get_ke_frac_new_bag[T: FloatOrArr](
        v_wall: T,
        alpha_n: float,
        n_xi: int = const.DEFAULT_N_XI,
        verbosity: int = 0) -> T:
    r"""
    Determine kinetic energy fraction (of total energy).
    Bag equation of state only so far, as it takes
    $e_n = \frac{3}{4} w_n (1 + \alpha_n)$.
    This assumes zero trace anomaly in broken phase.

    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param n_xi: number of $\xi$ points
    :param verbosity: logging verbosity
    :return: kinetic energy fraction
    """
    it = np.nditer([v_wall, None])
    for vw, ke in it:
        vw = vw.item()
        sol_type = identify_solution_type_bag(
            vw, alpha_n, df_dtau_ptr=DF_DTAU_PTR_BAG, ode_method=DEFAULT_FLUID_INTEGRATE_METHOD,
            cs2_fun=cs2_bag_scalar)
        if not sol_type == SolutionType.ERROR:
            # Now ready to solve for fluid profile
            v, w, xi = fluid_bag.sound_shell_bag(
                vw, alpha_n, cs2_fun_ptr=CS2_BAG_SCALAR_PTR, df_dtau_ptr=DF_DTAU_PTR_BAG,
                ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_fun=cs2_bag_scalar, n_xi=n_xi)
            ke[...] = kinetic_energy_density(v, w, xi, vw)
        else:
            ke[...] = np.nan
        if verbosity > 0:
            logger.debug(
                "%8.6f %8.6f %f",
                vw, alpha_n, ke
            )

    # Symmetric phase energy density
    e_s = bag.e_bag(w[-1], 0, bag.theta_bag(w[-1], 0, alpha_n))
    # result is stored in it.operands[1]
    ke_frac_out: T
    if isinstance(v_wall, np.ndarray):
        ke_frac_out = it.operands[1] / e_s
    else:
        ke_frac_out = type(v_wall)(it.operands[1]) / e_s

    return ke_frac_out


def _get_ubarf2_bag_scalar(
        v_wall: th.FloatOrArr1D,
        alpha_n: float,
        cs2_fun_ptr: th.CS2FunScalarPtr,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_fun: th.CS2Fun,
        n_xi: int = const.DEFAULT_N_XI,
        verbosity: int = 0,
        parallel: bool = True) -> float:
    if identify_solution_type_bag(
            v_wall, alpha_n, df_dtau_ptr=df_dtau_ptr, ode_method=ode_method,
            cs2_fun=cs2_fun) == SolutionType.ERROR:
        ub2 = np.nan
    else:
        # Now ready to solve for fluid profile
        v, w, xi = fluid_bag.sound_shell_bag(
            v_wall, alpha_n, cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr,
            ode_method=ode_method, cs2_fun=cs2_fun, n_xi=n_xi)
        ub2 = ubarf2(v, w, xi, v_wall, w_bar=w[-1])

    if verbosity > 0:
        with numba.objmode:
            logger.debug(
                "v_wall=%8.6f, alpha_n=%8.6f, ubarf2=%f",
                v_wall, alpha_n, ub2
            )
    return ub2


_get_ubarf2_bag_scalar_numba = njit(_get_ubarf2_bag_scalar, nogil=True)


def _get_ubarf2_bag_arr(
        v_wall: th.FloatArr1D,
        alpha_n: float,
        cs2_fun_ptr: th.CS2FunScalarPtr,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_fun: th.CS2Fun,
        n_xi: int = const.DEFAULT_N_XI,
        verbosity: int = 0) -> th.FloatArr1D:
    ubarf2 = np.zeros_like(v_wall)
    # pylint: disable=not-an-iterable
    for i in numba.prange(v_wall.size):
        ubarf2[i] = _get_ubarf2_bag_scalar_numba(
            v_wall[i], alpha_n, cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr,
            ode_method=ode_method, cs2_fun=cs2_fun, n_xi=n_xi, verbosity=verbosity)
    return ubarf2

_get_ubarf2_bag_arr_parallel, _get_ubarf2_bag_arr_single = njit_parallel_pair(_get_ubarf2_bag_arr, nogil=True)


def _get_ubarf2_bag_arr_wrapper(
        v_wall: th.FloatOrArr1D,
        alpha_n: float,
        cs2_fun_ptr: th.CS2FunScalarPtr,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_fun: th.CS2Fun,
        n_xi: int = const.DEFAULT_N_XI,
        verbosity: int = 0,
        parallel: bool = True) -> th.FloatArr1D:
    if parallel:
        return _get_ubarf2_bag_arr_parallel(
            v_wall, alpha_n, cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr,
            ode_method=ode_method, cs2_fun=cs2_fun, n_xi=n_xi, verbosity=verbosity)
    return _get_ubarf2_bag_arr_single(
        v_wall, alpha_n, cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr,
        ode_method=ode_method, cs2_fun=cs2_fun, n_xi=n_xi, verbosity=verbosity)


def get_ubarf2_bag[T: FloatOrArr1D](
        v_wall: T,
        alpha_n: float,
        cs2_fun_ptr: th.CS2FunScalarPtr,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_fun: th.CS2Fun,
        n_xi: int = const.DEFAULT_N_XI,
        verbosity: int = 0,
        parallel: bool = True) -> T:
    r"""
    Get mean square fluid velocity from $v_\text{wall}$ and $\alpha_n$.

    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param cs2_fun_ptr: pointer to the $c_s^2$ function
    :param df_dtau_ptr: pointer to the differential equation function
    :param ode_method: differential equation solver to be used
    :param cs2_fun: $c_s^2$ function
    :param n_xi: number of $\xi$ points
    :param verbosity: logging verbosity
    :param parallel: whether to compute the values for the elements of an array with multiple threads
    :return: mean square fluid velocity
    """
    if isinstance(v_wall, float):
        return _get_ubarf2_bag_scalar(  # type: ignore[return-value]
            v_wall, alpha_n, cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr,
            ode_method=ode_method, cs2_fun=cs2_fun, n_xi=n_xi, verbosity=verbosity)
    if isinstance(v_wall, np.ndarray):
        return _get_ubarf2_bag_arr(  # type: ignore[return-value]
            v_wall, alpha_n, cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr,
            ode_method=ode_method, cs2_fun=cs2_fun, n_xi=n_xi, verbosity=verbosity)
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


# The parallelism is selected at run time by _get_ubarf2_bag_arr_wrapper instead of with jit_options,
# so that the caller can disable it, e.g. when it's already running in parallel.
@overload(get_ubarf2_bag, jit_options={"nopython": True, "nogil": True})
def _get_ubarf2_bag_numba(
        v_wall: th.FloatOrArr1D,
        alpha_n: float,
        cs2_fun_ptr: th.CS2FunScalarPtr,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_fun: th.CS2Fun,
        n_xi: int = const.DEFAULT_N_XI,
        verbosity: int = 0,
        parallel: bool = True) -> th.NumbaFunc:
    if isinstance(v_wall, numba.types.Float):
        return _get_ubarf2_bag_scalar
    if isinstance(v_wall, numba.types.Array):
        if not v_wall.ndim:
            return _get_ubarf2_bag_scalar
        return _get_ubarf2_bag_arr_wrapper
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


def get_ubarf2_new_bag(
        v_wall: th.FloatOrArr,
        alpha_n: float,
        n_xi: int = const.DEFAULT_N_XI,
        verbosity: int = 0) -> th.FloatOrArr:
    r"""
    Get mean square fluid velocity from $v_\text{wall}$ and $\alpha_n$.

    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param n_xi: not used
    :param verbosity: logging verbosity
    :return: mean square fluid velocity
    """
    w_mean = 1  # For bag, it doesn't matter
    Gamma = bag.adiabatic_index_bag(w_mean, Phase.BROKEN, bag.theta_bag(w_mean, Phase.BROKEN, alpha_n))

    it = np.nditer([v_wall, None])
    for vw, Ubarf2 in it:
        vw = vw.item()
        sol_type = identify_solution_type_bag(
            vw, alpha_n, df_dtau_ptr=DF_DTAU_PTR_BAG, ode_method=DEFAULT_FLUID_INTEGRATE_METHOD,
            cs2_fun=cs2_bag_scalar)
        if not sol_type == SolutionType.ERROR:
            # Now ready to get Ubarf2
            ke_frac = get_ke_frac_new_bag(vw, alpha_n)
            Ubarf2[...] = ke_frac / Gamma
        else:
            Ubarf2[...] = np.nan
        if verbosity > 0:
            logger.debug(
                "%8.6f %8.6f %f",
                vw, alpha_n, Ubarf2
            )

    # Ubarf2 is stored in it.operands[1]
    ubarf2_out: th.FloatOrArr
    if isinstance(v_wall, np.ndarray):
        ubarf2_out = it.operands[1]
    else:
        ubarf2_out = type(v_wall)(it.operands[1])

    return ubarf2_out


def mean_energy_change_bag(
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float,
        alpha_n: float) -> float:
    r"""
    Bubble-averaged change in energy density in bubble relative to outside value.

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :return: mean energy change
    """
    #    def ene_diff(v,w,xi):
    #        return de_from_w(w, xi, v_wall, alpha_n)
    #    int1, int2 = split_integrate(ene_diff, v, w, xi**3, v_wall)
    #    integral = int1 + int2
    check.check_physical_params(
        (v_wall, alpha_n), df_dtau_ptr=DF_DTAU_PTR_BAG, ode_method=DEFAULT_FLUID_INTEGRATE_METHOD,
        cs2_fun=cs2_bag_scalar)
    integral = np.trapezoid(
        de_from_w_bag(
            w, xi, v_wall, alpha_n,
            df_dtau_ptr=DF_DTAU_PTR_BAG, ode_method=DEFAULT_FLUID_INTEGRATE_METHOD,
            cs2_fun=cs2_bag_scalar),
        xi ** 3)
    return integral / v_wall ** 3


def part_integrate(
        func: Integrand,
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        where_in: th.IntOrArr) -> float:
    r"""
    Integrate a function func of arrays $v, w, \xi$ over index selection where_in.

    :param func: function to be integrated
    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param where_in: index selection
    """
    xi_in = xi[where_in]
    v_in = v[where_in]
    w_in = w[where_in]
    integrand = func(v_in, w_in, xi_in)
    return np.trapezoid(integrand, xi_in)


def split_integrate(
        func: Integrand,
        v: th.FloatArr1D,
        w: th.FloatArr1D,
        xi: th.FloatArr1D,
        v_wall: float) -> tuple[float, float]:
    r"""
    Split an integration of a function func of arrays $v, w, \xi$
    according to whether $\xi$ is inside or outside the wall (expecting discontinuity there).

    :param func: function to be integrated
    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param v_wall: $v_\text{wall}$
    """
    check.check_wall_speed(v_wall)
    inside = np.where(xi < v_wall)
    outside = np.where(xi > v_wall)
    int1 = 0.
    int2 = 0.
    if v[inside].size >= 3:
        int1 = part_integrate(func, v, w, xi, inside)
    if v[outside].size >= 3:
        int2 = part_integrate(func, v, w, xi, outside)
    return int1, int2
