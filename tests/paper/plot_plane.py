"""
TODO: rename this to plot_plane_paper
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi

import pttools.type_hints as th
from pttools import bubble


def filter_not(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace the elements that are False in the mask with np.nan"""
    arr2 = arr.copy()
    arr2[np.logical_not(mask)] = np.nan
    return arr2


def get_label(rtol: float, atol: float) -> str:
    if not atol:
        return f"relative error > {rtol * 100:.2f} %"
    if not rtol:
        return f"absolute error > {atol:.1e}"
    return rf"error > {rtol:.1e}$\xi$ + {atol:.1e}"


def get_solver_name(method: th.ODESolver) -> str:
    if method == "numba_lsoda":
        return "numba"
    if method is spi.odeint:
        return "odeint"
    if isinstance(method, str):
        return method
    if isinstance(method, spi.OdeSolver):
        return method.__class__.__name__
    return method.__name__


def get_differing_inds(
        data: np.ndarray,
        data_ref: np.ndarray,
        i: int,
        rtol: float,
        atol: float = 0) -> np.ndarray:
    r"""Get the indices where $v$, $w$ or $\xi$ values do not meet the given tolerances"""
    # differing_b = np.isclose(data[:3, i, :], data_ref[:3, i, :], rtol=rtol, atol=atol)
    # differing_f = np.isclose(data[3:, i, :], data_ref[:3, i, :], rtol=rtol, atol=atol)
    data_v_b = data[0, i, :]
    data_xi_b = data[2, i, :]
    data_v_f = data[3, i, :]
    data_xi_f = data[5, i, :]
    differing_b = np.logical_not(np.logical_and(
        np.isclose(data_v_b, data_ref[0, i, :], rtol=rtol, atol=atol),
        np.isclose(data_xi_b, data_ref[2, i, :], rtol=rtol, atol=atol)
    ))
    differing_f = np.logical_not(np.logical_and(
        np.isclose(data_v_f, data_ref[3, i, :], rtol=rtol, atol=atol),
        np.isclose(data_xi_f, data_ref[5, i, :], rtol=rtol, atol=atol)
    ))
    return np.array([differing_b, differing_f])


def set_invalid_v_to_nan(v: np.ndarray):
    v[v < 0] = np.nan
    v[v > 1] = np.nan


def v_ahead_max(xi):
    """Maximum fluid velocity allowed"""
    return xi


def plot_v_excerpt(ax: plt.Axes, v_wall: float, alpha_plus: float, n_xi: int = 500):
    """Plots parts of solution obtained by integration of fluid equations.
    Supersonic deflagration solution comes in two parts, ahead and behind wall,
    each with about npts values."""
    v, w, xi = bubble.fluid_shell_alpha_plus(v_wall, alpha_plus, n_xi=n_xi)
    wall_type = bubble.identify_solution_type_bag(v_wall, alpha_plus)
    if wall_type != bubble.SolutionType.DETON:
        ahead = np.where((xi > v_wall) & (v > 0))
        xi_a = xi[ahead]
        v_a = v[ahead]
        ax.plot(xi_a, v_a, "b")
        ax.plot([xi_a[0]], [v_a[0]], "bo")
        ax.plot([xi_a[-1]], [v_a[-1]], "bo")
    if wall_type != bubble.SolutionType.SUB_DEF:
        behind = np.where((xi < v_wall) & (v > 0))
        xi_b = xi[behind]
        v_b = v[behind]
        ax.plot(xi_b, v_b, "b")
        ax.plot([xi_b[0]], [v_b[0]], "bo")
        ax.plot([xi_b[-1]], [v_b[-1]], "bo")


def plot_conditions(ax: plt.Axes, cs2_s: float, cs2_b: float):
    # Create a line v(xi) = xi to start solving on with forwards and backwards solutions
    # This is the maximum fluid speed ahead of wall for deflagrations
    xi_min = 0.0 + 1 / bubble.N_XI_DEFAULT
    xi_max = 1.0
    xi_line = np.linspace(xi_min, xi_max, bubble.N_XI_DEFAULT)
    va_max_line = v_ahead_max(xi_line)

    # Create the shock line for deflagrations
    v_shock_line = bubble.v_shock_bag(xi_line)
    v_shock_line[v_shock_line <= 0.0] = np.nan

    # Create a line to show the maximum (universe frame) fluid velocity behind the wall.
    # Relevant for detonations and supersonic deflagrations.
    vb_max_line = bubble.lorentz(xi_line, np.sqrt(cs2_b))
    vb_max_line[xi_line <= np.sqrt(cs2_b)] = np.nan

    # Plot lines for the wall conditions
    ax.plot(xi_line, va_max_line, 'k:', label=r'$v = \xi$')
    ax.plot(xi_line, v_shock_line, 'k--', label=r'$v = v_{\rm sh}(\xi)$')
    ax.plot(xi_line, vb_max_line, 'k-.', label=r'$v = \mu(\xi, c_s)$')


def plot_differing(
        ax: plt.Axes,
        data: np.ndarray,
        ref: np.ndarray,
        rtol_diffs: np.ndarray,
        atol_diffs: np.ndarray
        ):
    diff_small = get_differing_inds(data_b, deflag_ref, i, rtol=rtol_small_diff, atol=atol_small_diff)
    diff_mid = get_differing_inds(data_b, deflag_ref, i, rtol=rtol_mid_diff, atol=atol_mid_diff)
    diff_high = get_differing_inds(data_b, deflag_ref, i, rtol=rtol_high_diff, atol=atol_high_diff)
    diff_small[diff_mid] = 0
    diff_mid[diff_high] = 0

    for diff, color, rtol, atol in zip(
        (diff_small, diff_mid, diff_high),
        ("yellow", "orange", "red"),
        rtol_diffs,
        atol_diffs):
        label = get_label(rtol, atol) if not i else None
        ax.plot(
            filter_not(deflag_xi_b, diff[0, :]), filter_not(deflag_v_b, diff[0, :]),
            color=color, label=label)
        ax.plot(
            filter_not(deflag_xi_f, diff[1, :]), filter_not(deflag_v_f, diff[1, :]),
            color=color)


def plot_plane(
        ax: plt.Axes,
        data_s: np.ndarray,
        data_b: np.ndarray = None,
        method: th.ODESolver = None,
        deflag_ref: np.ndarray = None,
        rtol_small_diff: float = 1e-4,
        rtol_mid_diff: float = 1e-3,
        rtol_high_diff: float = 1e-2,
        atol_small_diff: float = 0,
        atol_mid_diff: float = 0,
        atol_high_diff: float = 0,
        tau_backwards_end: float = -100.0,
        cs2_s: float = bubble.const.CS0_2,
        cs2_b: float = bubble.const.CS0_2,
        selected_solutions: bool = True
    ):
    """
    Modified from
    `sound-shell-model/paper/python/fig_8r_xi-v_plane.py
    <https://bitbucket.org/hindmars/sound-shell-model/src/master/paper/python/fig_8r_xi-v_plane.py>`_.
    """
    # Define a suitable number of default lines to plot
    # n_xi0 = data_b.shape[1]
    # xi0_step = 1 / (n_xi0 + 1)
    # xi0_array = np.linspace(xi0_step, 1 - xi0_step, n_xi0)

    plot_conditions(ax, cs2_s, cs2_b)

    # n_xi = data_s.shape[2]
    # TODO: change this to something more reasonable
    for i in range(data_s.shape[1]):
        deflag_v_b = data_s[0, i, :]
        deflag_xi_b = data_s[2, i, :]
        deflag_v_f = data_s[3, i, :]
        deflag_xi_f = data_s[5, i, :]

        # Grey out parts of line which are unphysical
        unphysical = np.logical_and(
            deflag_v_b - bubble.v_shock_bag(deflag_xi_b) < 0,
            deflag_v_b - bubble.lorentz(deflag_xi_b, bubble.CS0) > 0)
        # But let's keep the unphysical points to look at
        deflag_v_b_grey = deflag_v_b[unphysical]
        deflag_xi_b_grey = deflag_xi_b[unphysical]
        deflag_v_b[unphysical] = np.nan
        # Unphysical doesn't quite work for last points, so ...
        deflag_v_b_grey[deflag_v_b_grey < 1e-4] = np.nan

        if method == "numba_lsoda":
            set_invalid_v_to_nan(deflag_v_b)
            set_invalid_v_to_nan(deflag_v_f)
            set_invalid_v_to_nan(deflag_v_b_grey)

        # Plot
        grey = (0.8, 0.8, 0.8)
        ax.plot(deflag_xi_f, deflag_v_f, color=grey)
        ax.plot(deflag_xi_b, deflag_v_b, 'k')
        ax.plot(deflag_xi_b_grey, deflag_v_b_grey, color=grey)

        # if deflag_ref is not None:
            # TODO
            # plot_differing()

        # Make and plot a few lines starting from xi = 1
        # This code seems not to be used, so it's disabled.
        # if not i % 2:
        #     det_v_b, det_w_b, det_xi_b, _ = \
        #         bubble.fluid_integrate_param(xi0, 1, 1, t_end=tau_backwards_end, n_xi=n_xi)

    if data_b is not None:
        for i in range(data_b.shape[1]):
            det_v = data_b[0, i, :]
            det_xi = data_b[2, i, :]
            ax.plot(det_xi, det_v, "k")

    # Plot curves corresponding to selected solutions (c.f. Espinosa et al 2010)
    if selected_solutions:
        plot_v_excerpt(ax, 0.5, 0.263)
        plot_v_excerpt(ax, 0.7, 0.052)
        plot_v_excerpt(ax, 0.77, 0.091)

    # Make plot look nice
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r'$\xi$')
    ax.set_ylabel(r'$v(\xi)$')
    ax.grid()
    ax.legend(loc='upper left')
    if method:
        ax.set_title(get_solver_name(method))
