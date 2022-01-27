import numpy as np

from pttools import bubble
from pttools import speedup


def xiv_plane(
        method: str = "odeint",
        tau_forwards_end: float = 100.0,
        tau_backwards_end: float = -100.0,
        n_xi0: int = 9,
        n_xi: int = 1000,
        df_dtau_ptr: speedup.DifferentialPointer = bubble.fluid.DF_DTAU_BAG_PTR,
        cs2_s=bubble.CS0_2,
        cs2_b=bubble.CS0_2
    ) -> np.ndarray:
    """
    Modified from :ssm_repo:`paper/python/fig_8r_xi-v_plane.py`
    """
    # Define a suitable number of default lines to plot
    xi0_step = 1 / (n_xi0 + 1)
    xi0_array = np.linspace(xi0_step, 1 - xi0_step, n_xi0)

    deflag = np.zeros((6, len(xi0_array), n_xi))
    for i, xi0 in enumerate(xi0_array):
        # Make lines starting from v = xi, forward and back
        # Curves below the v=xi line
        deflag_v_b, deflag_w_b, deflag_xi_b, _ = bubble.fluid_integrate_param(
            v0=xi0, w0=1, xi0=xi0, t_end=tau_backwards_end, n_xi=n_xi, df_dtau_ptr=df_dtau_ptr, method=method)
        # Curves above the v=xi line
        deflag_v_f, deflag_w_f, deflag_xi_f, _ = bubble.fluid_integrate_param(
            v0=xi0, w0=1, xi0=xi0, t_end=tau_forwards_end, n_xi=n_xi, df_dtau_ptr=df_dtau_ptr, method=method)
        # Filter out the unphysical part of the curves
        unphysical = np.logical_and(
            deflag_v_b < bubble.v_shock(deflag_xi_b, cs2=cs2_s),
            deflag_v_b > bubble.lorentz(deflag_xi_b, np.sqrt(cs2_b))
        )
        deflag_v_b[unphysical] = np.nan

        deflag[:, i, :] = [deflag_v_b, deflag_w_b, deflag_xi_b, deflag_v_f, deflag_w_f, deflag_xi_f]
    return deflag
