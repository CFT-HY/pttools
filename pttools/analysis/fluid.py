import numpy as np

from pttools.bubble import \
    DEFAULT_FLUID_INTEGRATE_METHOD, DF_DTAU_PTR_BAG, \
    DifferentialPointer, FluidIntegrateMethod, Phase, \
    fluid_integrate_param, v_max_behind
from pttools.type_hints import FloatArr1D, FloatArr3D

DEFAULT_CURVES_BROKEN_V = np.linspace(0, 1, 5)
DEFAULT_CURVES_INVERSE_V = np.linspace(-1, 0, 10)
DEFAULT_CURVES_SYMMETRIC_V = np.linspace(0, 1, 10)


def curves_broken(
        v0: FloatArr1D = DEFAULT_CURVES_BROKEN_V,
        w0: float = 1.,
        df_dtau_ptr: DifferentialPointer = DF_DTAU_PTR_BAG,
        method: FluidIntegrateMethod = DEFAULT_FLUID_INTEGRATE_METHOD,
        n_xi: int = 1000,
        tau_end: float = -100.) -> FloatArr3D:
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
        tau_end: float = -100.):
    return curves_inverse(v0=v0, w0=w0, xi0=-1., df_dtau_ptr=df_dtau_ptr, method=method, n_xi=n_xi, tau_end=tau_end)


def curves_inverse(
        v0: FloatArr1D = DEFAULT_CURVES_BROKEN_V,
        w0: float = 1.,
        xi0: float = 1.,
        df_dtau_ptr: DifferentialPointer = DF_DTAU_PTR_BAG,
        method: FluidIntegrateMethod = DEFAULT_FLUID_INTEGRATE_METHOD,
        n_xi: int = 1000,
        tau_end: float = -100.) -> FloatArr3D:
    # Todo: implement a cut for the phases
    return curves_broken(v0=v0, w0=w0, df_dtau_ptr=df_dtau_ptr, method=method, n_xi=n_xi, tau_end=tau_end)


def curves_symmetric(
        v0: FloatArr1D = DEFAULT_CURVES_SYMMETRIC_V,
        csb: float = None,
        w0: float = 1.,
        df_dtau_ptr: DifferentialPointer = DF_DTAU_PTR_BAG,
        method: FluidIntegrateMethod = DEFAULT_FLUID_INTEGRATE_METHOD,
        n_xi: int = 1000,
        tau_end_backwards: float = -100.,
        tau_end_forwards: float = 100.) -> FloatArr3D:
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
            unphysical = v_max_behind(xi=xi_b, csb=csb)
            v_b[unphysical] = np.nan

        data[:, i, :n_xi] = [v_b, w_b, xi_b]
        data[:, i, n_xi:] = [v_f, w_f, xi_f]

    return data
