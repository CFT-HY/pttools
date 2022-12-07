"""Functions for calculating quantities derived from solutions

TODO: Should this be renamed as thermodynamics?
"""

import logging

import typing as tp

import numba
import numpy as np

import pttools.type_hints as th
from . import bag
from . import boundary
from . import check
from . import const
from . import fluid
from . import relativity
from . import transition

Integrand = tp.Union[
    tp.Callable[
        [np.ndarray, np.ndarray, np.ndarray],
        np.ndarray
    ],
    tp.Callable[
        [float, float, float],
        float
    ]
]

logger = logging.getLogger(__name__)


@numba.njit
def de_from_w_bag(w: np.ndarray, xi: np.ndarray, v_wall: float, alpha_n: float) -> np.ndarray:
    r"""
    Calculates energy density difference ``de = e - e[-1]`` from enthalpy, assuming
    bag equation of state.
    Can get ``alpha_n = find_alpha_n_from_w_xi(w,xi,v_wall,alpha_p)``

    :param w: $w$
    :param xi: $\xi$
    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :return: energy density difference de
    """
    check.check_physical_params((v_wall, alpha_n))
    e_from_w = bag.e_bag(w, boundary.get_phase(xi, v_wall), 0.75 * w[-1] * alpha_n)

    return e_from_w - e_from_w[-1]


@numba.njit
def de_from_w_new_bag(v: np.ndarray, w: np.ndarray, xi: np.ndarray, v_wall: float, alpha_n: float) -> np.ndarray:
    r"""
    For exploring new methods of calculating energy density difference
    from velocity and enthalpy, assuming bag equation of state.

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :return: energy density difference de
    """
    check.check_physical_params((v_wall, alpha_n))
    e_from_w = bag.e_bag(w, boundary.get_phase(xi, v_wall), 0.75 * w[-1] * alpha_n)

    de = e_from_w - e_from_w[-1]

    # Try adjusting by a factor - currently doesn't do anything
    de *= 1.0

    return de


def get_kappa(
        v_wall: th.FloatOrArr,
        alpha_n: float,
        n_xi: int = const.N_XI_DEFAULT,
        verbosity: int = 0) -> th.FloatOrArr:
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
        sol_type = transition.identify_solution_type_bag(vw, alpha_n)

        if not sol_type == boundary.SolutionType.ERROR:
            # Now ready to solve for fluid profile
            v, w, xi = fluid.fluid_shell(vw, alpha_n, n_xi)

            kappa[...] = ubarf_squared(v, w, xi, vw) / (0.75 * alpha_n)
        else:
            kappa[...] = np.nan
        if verbosity > 0:
            logger.debug(f"{vw:8.6f} {alpha_n:8.6f} {kappa}")

    if isinstance(v_wall, np.ndarray):
        kappa_out = it.operands[1]
    else:
        kappa_out = type(v_wall)(it.operands[1])

    return kappa_out


def get_kappa_de(
        v_wall: th.FloatOrArr,
        alpha_n: float,
        n_xi: int = const.N_XI_DEFAULT,
        verbosity: int = 0) -> tp.Union[tp.Tuple[float, float], tp.Tuple[np.ndarray, np.ndarray]]:
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
        sol_type = transition.identify_solution_type_bag(vw, alpha_n)

        if not sol_type == boundary.SolutionType.ERROR:
            # Now ready to solve for fluid profile
            v, w, xi = fluid.fluid_shell(vw, alpha_n, n_xi)
            # Esp+ epsilon is alpha_n * 0.75*w_n
            kappa[...] = (4 / 3) * ubarf_squared(v, w, xi, vw)
            de[...] = mean_energy_change_bag(v, w, xi, vw, alpha_n)
        else:
            kappa[...] = np.nan
            de[...] = np.nan
        if verbosity > 0:
            logger.debug(f"{vw:8.6f} {alpha_n:8.6f} {kappa} {de}")

    if isinstance(v_wall, np.ndarray):
        kappa_out = it.operands[1]
        de_out = it.operands[2]
    else:
        kappa_out = type(v_wall)(it.operands[1])
        de_out = type(v_wall)(it.operands[2])

    return kappa_out, de_out


def get_kappa_dq(
        v_wall: th.FloatOrArr,
        alpha_n: float,
        n_xi: int = const.N_XI_DEFAULT,
        verbosity: int = 0) -> tp.Union[tp.Tuple[float, float], tp.Tuple[np.ndarray, np.ndarray]]:
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
        sol_type = transition.identify_solution_type_bag(vw, alpha_n)

        if not sol_type == boundary.SolutionType.ERROR:
            # Now ready to solve for fluid profile
            v, w, xi = fluid.fluid_shell(vw, alpha_n, n_xi)
            # Esp+ epsilon is alpha_n * 0.75*w_n
            kappa[...] = ubarf_squared(v, w, xi, vw) / (0.75 * alpha_n)
            dq[...] = 0.75 * mean_enthalpy_change(v, w, xi, vw) / (0.75 * alpha_n * w[-1])
        else:
            kappa[...] = np.nan
            dq[...] = np.nan
        if verbosity > 0:
            logger.debug(f"{vw:8.6f} {alpha_n:8.6f} {kappa} {dq}")

    if isinstance(v_wall, np.ndarray):
        kappa_out = it.operands[1]
        dq_out = it.operands[2]
    else:
        kappa_out = type(v_wall)(it.operands[1])
        dq_out = type(v_wall)(it.operands[2])

    return kappa_out, dq_out


def get_ke_de_frac_bag(
        v_wall: th.FloatOrArr,
        alpha_n: float,
        n_xi: int = const.N_XI_DEFAULT,
        verbosity: int = 0) -> tp.Union[tp.Tuple[float, float], tp.Tuple[np.ndarray, np.ndarray]]:
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
        sol_type = transition.identify_solution_type_bag(vw, alpha_n)

        if not sol_type == boundary.SolutionType.ERROR:
            # Now ready to solve for fluid profile
            v, w, xi = fluid.fluid_shell(vw, alpha_n, n_xi)
            # Esp+ epsilon is alpha_n * 0.75*w_n
            ke[...] = ubarf_squared(v, w, xi, vw) / (0.75 * (1 + alpha_n))
            de[...] = mean_energy_change_bag(v, w, xi, vw, alpha_n) / (0.75 * w[-1] * (1 + alpha_n))
        else:
            ke[...] = np.nan
            de[...] = np.nan
        if verbosity > 0:
            logger.debug(f"{vw:8.6f} {alpha_n:8.6f} {ke} {de}")

    if isinstance(v_wall, np.ndarray):
        ke_out = it.operands[1]
        de_out = it.operands[2]
    else:
        ke_out = type(v_wall)(it.operands[1])
        de_out = type(v_wall)(it.operands[2])

    return ke_out, de_out


def get_ke_frac_bag(v_wall: th.FloatOrArr, alpha_n: float, n_xi: int = const.N_XI_DEFAULT) -> th.FloatOrArr:
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
    ubar2 = get_ubarf2(v_wall, alpha_n, n_xi)
    return ubar2 / (0.75 * (1 + alpha_n))


def get_ke_frac_new_bag(
        v_wall: th.FloatOrArr,
        alpha_n: float,
        n_xi: int = const.N_XI_DEFAULT,
        verbosity: int = 0) -> th.FloatOrArr:
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
        sol_type = transition.identify_solution_type_bag(vw, alpha_n)
        if not sol_type == boundary.SolutionType.ERROR:
            # Now ready to solve for fluid profile
            v, w, xi = fluid.fluid_shell(vw, alpha_n, n_xi)
            ke[...] = mean_kinetic_energy(v, w, xi, vw)
        else:
            ke[...] = np.nan
        if verbosity > 0:
            logger.debug(f"{vw:8.6f} {alpha_n:8.6f} {ke}")

    # Symmetric phase energy density
    e_s = bag.e_bag(w[-1], 0, bag.theta_bag(w[-1], 0, alpha_n))
    # result is stored in it.operands[1]
    if isinstance(v_wall, np.ndarray):
        ke_frac_out = it.operands[1] / e_s
    else:
        ke_frac_out = type(v_wall)(it.operands[1]) / e_s

    return ke_frac_out


@numba.njit
def _get_ubarf2_scalar(v_wall: float, alpha_n: float, n_xi: int, verbosity: int) -> float:
    if transition.identify_solution_type_bag(v_wall, alpha_n) == boundary.SolutionType.ERROR:
        ubarf2 = np.nan
    else:
        # Now ready to solve for fluid profile
        v, w, xi = fluid.fluid_shell(v_wall, alpha_n, n_xi)
        ubarf2 = ubarf_squared(v, w, xi, v_wall)

    if verbosity > 0:
        with numba.objmode:
            logger.debug(f"v_wall=%8.6f, alpha_n=%8.6f, ubarf2=%f", v_wall, alpha_n, ubarf2)
    return ubarf2


@numba.njit(parallel=True)
def _get_ubarf2_arr(v_wall: np.ndarray, alpha_n: float, n_xi: int, verbosity: int) -> np.ndarray:
    ubarf2 = np.zeros_like(v_wall)
    for i in numba.prange(v_wall.size):
        ubarf2[i] = _get_ubarf2_scalar(v_wall[i], alpha_n, n_xi, verbosity)
    return ubarf2


@numba.generated_jit(nopython=True)
def get_ubarf2(
        v_wall: th.FloatOrArr,
        alpha_n: float,
        n_xi: int = const.N_XI_DEFAULT,
        verbosity: int = 0) -> th.FloatOrArr:
    r"""
    Get mean square fluid velocity from $v_\text{wall}$ and $\alpha_n$.

    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param n_xi: number of $\xi$ points
    :param verbosity: logging verbosity
    :return: mean square fluid velocity
    """
    if isinstance(v_wall, numba.types.Float):
        return _get_ubarf2_scalar
    if isinstance(v_wall, numba.types.Array):
        if not v_wall.ndim:
            return _get_ubarf2_scalar
        return _get_ubarf2_arr
    if isinstance(v_wall, float):
        return _get_ubarf2_scalar(v_wall, alpha_n, n_xi, verbosity)
    if isinstance(v_wall, np.ndarray):
        return _get_ubarf2_arr(v_wall, alpha_n, n_xi, verbosity)
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


def get_ubarf2_new_bag(
        v_wall: th.FloatOrArr,
        alpha_n: float,
        n_xi: int = const.N_XI_DEFAULT,
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
    Gamma = bag.adiabatic_index_bag(w_mean, const.BROK_PHASE, bag.theta_bag(w_mean, const.BROK_PHASE, alpha_n))
    logger.debug(Gamma)

    it = np.nditer([v_wall, None])
    for vw, Ubarf2 in it:
        sol_type = transition.identify_solution_type_bag(vw, alpha_n)
        if not sol_type == boundary.SolutionType.ERROR:
            # Now ready to get Ubarf2
            ke_frac = get_ke_frac_new_bag(vw, alpha_n)
            Ubarf2[...] = ke_frac / Gamma
        else:
            Ubarf2[...] = np.nan
        if verbosity > 0:
            logger.debug(f"{vw:8.6f} {alpha_n:8.6f} {Ubarf2}")

    # Ubarf2 is stored in it.operands[1]
    if isinstance(v_wall, np.ndarray):
        ubarf2_out = it.operands[1]
    else:
        ubarf2_out = type(v_wall)(it.operands[1])

    return ubarf2_out


def mean_energy_change_bag(v: np.ndarray, w: np.ndarray, xi: np.ndarray, v_wall: float, alpha_n: float) -> float:
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
    check.check_physical_params((v_wall, alpha_n))
    integral = np.trapz(de_from_w_bag(w, xi, v_wall, alpha_n), xi ** 3)
    return integral / v_wall ** 3


def mean_enthalpy_change(v: np.ndarray, w: np.ndarray, xi: np.ndarray, v_wall: float) -> float:
    r"""
    Mean change in enthalpy in bubble relative to outside value.

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param v_wall: $v_\text{wall}$
    :return: mean enthalpy change
    """
    #    def en_diff(v, dw, xi):
    #        return dw
    #    int1, int2 = split_integrate(en_diff, v, w - w[-1], xi**3, v_wall)
    #    integral = int1 + int2
    check.check_wall_speed(v_wall)
    integral = np.trapz((w - w[-1]), xi ** 3)
    return integral / v_wall ** 3


@numba.njit
def mean_kinetic_energy(v: np.ndarray, w: np.ndarray, xi: np.ndarray, v_wall: float) -> float:
    r"""
    Kinetic energy of fluid in bubble, averaged over bubble volume,
    from fluid shell functions.

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param v_wall: $v_\text{wall}$
    :return: mean kinetic energy
    """
    check.check_wall_speed(v_wall)
    integral = np.trapz(w * v ** 2 * relativity.gamma2(v), xi ** 3)
    return integral / (v_wall ** 3)


def part_integrate(
        func: Integrand,
        v: np.ndarray,
        w: np.ndarray,
        xi: np.ndarray,
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
    return np.trapz(integrand, xi_in)


def split_integrate(
        func: Integrand,
        v: np.ndarray,
        w: np.ndarray,
        xi: np.ndarray,
        v_wall: float) -> tp.Tuple[float, float]:
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


@numba.njit
def ubarf_squared(v: np.ndarray, w: np.ndarray, xi: np.ndarray, v_wall: float) -> float:
    r"""
    Enthalpy-weighted mean square space components of 4-velocity of fluid in bubble,
    from fluid shell functions.

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param v_wall: $v_\text{wall}$
    """
    check.check_wall_speed(v_wall)
    #    def fun(v,w,xi):
    #        return w * v**2 * gamma2(v)
    #    int1, int2 = split_integrate(fun, v, w, xi**3, v_wall)
    #    integral = int1 + int2
    #    integral = np.trapz(w * v**2 * gamma2(v), xi**3)

    return mean_kinetic_energy(v, w, xi, v_wall) / w[-1]
