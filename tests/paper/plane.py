import numpy as np

from pttools import bubble


def xiv_plane(
        method: str = "odeint",
        tau_forwards_end: float = 100.0,
        tau_backwards_end: float = -100.0,
        n_xi0: int = 9,
        n_xi: int = 1000) -> np.ndarray:
    """
    Modified from :ssm_repo:`paper/python/fig_8r_xi-v_plane.py`
    """
    # Define a suitable number of default lines to plot
    xi0_step = 1 / (n_xi0 + 1)
    xi0_array = np.linspace(xi0_step, 1 - xi0_step, n_xi0)

    deflag = np.zeros((6, len(xi0_array), n_xi))
    for i, xi0 in enumerate(xi0_array):
        # Make lines starting from v = xi, forward and back
        deflag_v_b, deflag_w_b, deflag_xi_b, _ = bubble.fluid_integrate_param(
            xi0, 1, xi0, t_end=tau_backwards_end, n_xi=n_xi, method=method)
        deflag_v_f, deflag_w_f, deflag_xi_f, _ = bubble.fluid_integrate_param(
            xi0, 1, xi0, t_end=tau_forwards_end, n_xi=n_xi, method=method)
        # Grey out parts of line which are unphysical
        unphysical = np.logical_and(
            deflag_v_b - bubble.v_shock(deflag_xi_b) < 0,
            deflag_v_b - bubble.lorentz(deflag_xi_b, bubble.CS0) > 0)
        # But let"s keep the unphysical points to look at
        deflag_v_b[unphysical] = np.nan

        deflag[:, i, :] = [deflag_v_b, deflag_w_b, deflag_xi_b, deflag_v_f, deflag_w_f, deflag_xi_f]
    return deflag
