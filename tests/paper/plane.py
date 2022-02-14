import typing as tp

import numpy as np

from pttools import bubble
from pttools import speedup


def xiv_plane(
        method: str = "odeint",
        tau_forwards_end: float = 100.0,
        tau_backwards_end: float = -100.0,
        n_xi0_b: int = 6,
        n_xi0_s: int = 9,
        n_xi: int = 1000,
        df_dtau_ptr: speedup.DifferentialPointer = bubble.fluid.DF_DTAU_BAG_PTR,
        cs2_s=bubble.CS0_2,
        cs2_b=bubble.CS0_2,
        separate_phases: bool = True
    ) -> tp.Union[tp.Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Modified from :ssm_repo:`paper/python/fig_8r_xi-v_plane.py`
    """
    # Define a suitable number of default lines to plot
    xi0_step_b = (1 - np.sqrt(cs2_b)) / (n_xi0_b + 1)
    xi0_step_s = 1 / (n_xi0_s + 1)
    xi0_array_b = np.linspace(np.sqrt(cs2_b) + xi0_step_b, 1 - xi0_step_b, n_xi0_b)
    xi0_array_s = np.linspace(xi0_step_s, 1 - xi0_step_s, n_xi0_s)

    data_s = np.zeros((6, xi0_array_s.size, n_xi))
    data_b = np.zeros((3, xi0_array_b.size, n_xi))
    # Symmetric phase: deflagrations
    for i, xi0 in enumerate(xi0_array_s):
        # Make lines starting from v = xi, forward and back
        # Curves below the v=xi line
        deflag_v_b, deflag_w_b, deflag_xi_b, _ = bubble.fluid_integrate_param(
            v0=xi0, w0=1, xi0=xi0,
            t_end=tau_backwards_end, n_xi=n_xi, df_dtau_ptr=df_dtau_ptr, method=method, phase=bubble.Phase.SYMMETRIC)
        # Curves above the v=xi line
        deflag_v_f, deflag_w_f, deflag_xi_f, _ = bubble.fluid_integrate_param(
            v0=xi0, w0=1, xi0=xi0,
            t_end=tau_forwards_end, n_xi=n_xi, df_dtau_ptr=df_dtau_ptr, method=method, phase=bubble.Phase.SYMMETRIC)
        # Filter out the unphysical part of the curves
        if separate_phases:
            deflag_v_b[deflag_v_b < bubble.v_shock(deflag_xi_b, cs2=cs2_s)] = np.nan
        else:
            unphysical = np.logical_and(
                deflag_v_b < bubble.v_shock(deflag_xi_b, cs2=cs2_s),
                deflag_v_b > bubble.lorentz(deflag_xi_b, np.sqrt(cs2_b))
            )
            deflag_v_b[unphysical] = np.nan

        data_s[:, i, :] = [deflag_v_b, deflag_w_b, deflag_xi_b, deflag_v_f, deflag_w_f, deflag_xi_f]

    if not separate_phases:
        return data_s

    # Broken phase: detonations
    for i, xi0 in enumerate(xi0_array_b):
        v_shock = bubble.lorentz(xi0, np.sqrt(cs2_b))
        if v_shock > 0:
            det_v, det_w, det_xi, _ = bubble.fluid_integrate_param(
                v0=v_shock, w0=1, xi0=xi0,
                t_end=tau_backwards_end, n_xi=n_xi, df_dtau_ptr=df_dtau_ptr, method=method, phase=bubble.Phase.BROKEN)
            data_b[:, i, :] = [det_v, det_w, det_xi]
        else:
            data_b[:, i, :] = np.nan

    return data_b, data_s
