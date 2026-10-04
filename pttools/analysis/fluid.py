"""Utilities for computing the fluid profile curves that are drawn on the plots."""

import numpy as np

from pttools.bubble import (
    DEFAULT_FLUID_INTEGRATE_METHOD,
    DF_DTAU_PTR_BAG,
    DifferentialPointer,
    FluidIntegrateMethod,
    Phase,
    fluid_integrate_param,
    v_max_behind,
)
from pttools.type_hints import FloatArr1D, FloatArr3D

DEFAULT_CURVES_BROKEN_V: FloatArr1D = np.linspace(0, 1, 5)
DEFAULT_CURVES_INVERSE_V: FloatArr1D = np.linspace(-1, 0, 10)
# The end points are excluded, as the integration cannot start from $v = \xi = 0$.
DEFAULT_CURVES_SYMMETRIC_V: FloatArr1D = np.linspace(0, 1, 12)[1:-1]


def curves_broken(
        v0: FloatArr1D = DEFAULT_CURVES_BROKEN_V,
        w0: float = 1.,
        df_dtau_ptr: DifferentialPointer = DF_DTAU_PTR_BAG,
        method: FluidIntegrateMethod = DEFAULT_FLUID_INTEGRATE_METHOD,
        n_xi: int = 1000,
        tau_end: float = -100.) -> FloatArr3D:
    r"""Fluid profile curves in the broken phase, integrated backwards from $\xi_0 = 1$.

    :param v0: $v_0$, starting fluid velocities of the curves
    :param w0: $w_0$, starting enthalpy density
    :param df_dtau_ptr: pointer to the differential equation function
    :param method: differential equation solver to be used
    :param n_xi: number of $\xi$ points per curve
    :param tau_end: $\tau_\text{end}$, end value of the integration parameter
    :return: array of $v, w, \xi$ with the shape (3, v0.size, n_xi)
    """
    data = np.zeros((3, v0.size, n_xi))

    for i, v0_i in enumerate(v0):
        v, w, xi, _ = fluid_integrate_param(
            v0=v0_i, w0=w0, xi0=1.,
            t_end=tau_end, n_xi=n_xi, df_dtau_ptr=df_dtau_ptr, method=method, phase=Phase.BROKEN
        )
        data[:, i, :] = [v, w, xi]

    return data


def curves_droplet(
        v0: FloatArr1D = -DEFAULT_CURVES_INVERSE_V,
        w0: float = 1.,
        df_dtau_ptr: DifferentialPointer = DF_DTAU_PTR_BAG,
        method: FluidIntegrateMethod = DEFAULT_FLUID_INTEGRATE_METHOD,
        n_xi: int = 1000,
        tau_end: float = -100.) -> FloatArr3D:
    r"""Fluid profile curves for droplets, i.e. inverse phase transitions.

    This calls :func:`curves_inverse` with $\xi_0 = -1$.

    :param v0: $v_0$, starting fluid velocities of the curves
    :param w0: $w_0$, starting enthalpy density
    :param df_dtau_ptr: pointer to the differential equation function
    :param method: differential equation solver to be used
    :param n_xi: number of $\xi$ points per curve
    :param tau_end: $\tau_\text{end}$, end value of the integration parameter
    :return: array of $v, w, \xi$ with the shape (3, v0.size, n_xi)
    """
    return curves_inverse(v0=v0, w0=w0, xi0=-1., df_dtau_ptr=df_dtau_ptr, method=method, n_xi=n_xi, tau_end=tau_end)


def curves_inverse(
        v0: FloatArr1D = DEFAULT_CURVES_BROKEN_V,
        w0: float = 1.,
        xi0: float = 1.,
        df_dtau_ptr: DifferentialPointer = DF_DTAU_PTR_BAG,
        method: FluidIntegrateMethod = DEFAULT_FLUID_INTEGRATE_METHOD,
        n_xi: int = 1000,
        tau_end: float = -100.) -> FloatArr3D:
    r"""Fluid profile curves for inverse phase transitions.

    The cut for the phases and the starting point $\xi_0$ are not yet implemented,
    and therefore this currently returns the same curves as :func:`curves_broken`.

    :param v0: $v_0$, starting fluid velocities of the curves
    :param w0: $w_0$, starting enthalpy density
    :param xi0: $\xi_0$, starting point (not yet used)
    :param df_dtau_ptr: pointer to the differential equation function
    :param method: differential equation solver to be used
    :param n_xi: number of $\xi$ points per curve
    :param tau_end: $\tau_\text{end}$, end value of the integration parameter
    :return: array of $v, w, \xi$ with the shape (3, v0.size, n_xi)
    """
    # Todo: implement a cut for the phases
    return curves_broken(v0=v0, w0=w0, df_dtau_ptr=df_dtau_ptr, method=method, n_xi=n_xi, tau_end=tau_end)


def curves_symmetric(
        v0: FloatArr1D = DEFAULT_CURVES_SYMMETRIC_V,
        csb: float | None = None,
        w0: float = 1.,
        df_dtau_ptr: DifferentialPointer = DF_DTAU_PTR_BAG,
        method: FluidIntegrateMethod = DEFAULT_FLUID_INTEGRATE_METHOD,
        n_xi: int = 1000,
        tau_end_backwards: float = -100.,
        tau_end_forwards: float = 100.) -> FloatArr3D:
    r"""Fluid profile curves in the symmetric phase, starting from the $v = \xi$ line.

    The curves are integrated both backwards (below the $v = \xi$ line) and forwards (above it).

    :param v0: $v_0 = \xi_0$, starting fluid velocities of the curves
    :param csb: $c_{s,b}$, sound speed in the broken phase.
        If given, the part of the backwards curves below the $\mu(\xi, v) = c_{s,b}$ curve is removed.
    :param w0: $w_0$, starting enthalpy density
    :param df_dtau_ptr: pointer to the differential equation function
    :param method: differential equation solver to be used
    :param n_xi: number of $\xi$ points per curve and direction
    :param tau_end_backwards: $\tau_\text{end}$ for the backwards integration
    :param tau_end_forwards: $\tau_\text{end}$ for the forwards integration
    :return: array of $v, w, \xi$ with the shape (3, v0.size, 2*n_xi)
    """
    data = np.empty((3, v0.size, 2 * n_xi))

    for i, v0_i in enumerate(v0):
        # Curves below the v=xi line
        v_b, w_b, xi_b, _ = fluid_integrate_param(
            v0=v0_i, w0=w0, xi0=v0_i,
            t_end=tau_end_backwards, n_xi=n_xi, df_dtau_ptr=df_dtau_ptr, method=method, phase=Phase.SYMMETRIC
        )
        # Curves above the v=xi line
        v_f, w_f, xi_f, _ = fluid_integrate_param(
            v0=v0_i, w0=w0, xi0=v0_i,
            t_end=tau_end_forwards, n_xi=n_xi, df_dtau_ptr=df_dtau_ptr, method=method, phase=Phase.SYMMETRIC
        )
        # Remove the part of the curves below the mu curve
        if csb is not None:
            v_b[v_b < v_max_behind(xi=xi_b, cs=csb)] = np.nan

        data[:, i, :n_xi] = [v_b, w_b, xi_b]
        data[:, i, n_xi:] = [v_f, w_f, xi_f]

    return data
