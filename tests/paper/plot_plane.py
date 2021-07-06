import matplotlib.pyplot as plt
import numpy as np

from pttools import bubble


def v_ahead_max(xi):
    """Maximum fluid velocity allowed"""
    return xi


def plot_v_excerpt(v_wall, alpha_plus):
    """Plots parts of solution obtained by integration of fluid equations.
    Supersonic deflgration solution comes in two parts, ahead and behind wall,
    each with about npts values."""
    npts = 500
    v, w, xi = bubble.fluid_shell_alpha_plus(v_wall, alpha_plus, n_xi=npts)
    # lenv = len(v)
    # print(lenv)
    wall_type = bubble.identify_solution_type(v_wall, alpha_plus)
    if wall_type != bubble.SolutionType.DETON:
        ahead = np.where((xi > v_wall) & (v > 0))
        xi_a = xi[ahead]
        v_a = v[ahead]
        plt.plot(xi_a, v_a, "b")
        plt.plot([xi_a[0]], [v_a[0]], "bo")
        plt.plot([xi_a[-1]], [v_a[-1]], "bo")
    if wall_type != bubble.SolutionType.SUB_DEF:
        behind = np.where((xi < v_wall) & (v > 0))
        xi_b = xi[behind]
        v_b = v[behind]
        plt.plot(xi_b, v_b, "b")
        plt.plot([xi_b[0]], [v_b[0]], "bo")
        plt.plot([xi_b[-1]], [v_b[-1]], "bo")
    return 1


def plot_plane(deflag: np.ndarray, tau_backwards_end: float = -100.0):
    """Modified from sound-shell-model/paper/python/fig_8r_xi-v_plane.py"""
    # Define a suitable number of default lines to plot
    n_xi0 = deflag.shape[1]
    xi0_step = 1 / (n_xi0 + 1)
    xi0_array = np.linspace(xi0_step, 1 - xi0_step, n_xi0)

    # Create a line v(xi) = xi to start solving on with forwards and backwards solutions
    # This is the maximum fluid speed ahead of wall for deflagrations
    xi_min = 0.0 + 1 / bubble.N_XI_DEFAULT
    xi_max = 1.0
    xi_line = np.linspace(xi_min, xi_max, bubble.N_XI_DEFAULT)
    va_max_line = v_ahead_max(xi_line)

    # Create the shock line for deflagrations
    v_shock_line = bubble.v_shock(xi_line)
    v_shock_line[v_shock_line <= 0.0] = np.nan

    # Create a line to show the maximum (universe frame) fluid velocity behind the wall.
    # Relevant for detonations and supersonic deflagrations.
    vb_max_line = bubble.lorentz(xi_line, bubble.CS0)
    vb_max_line[xi_line <= bubble.CS0] = np.nan

    # Plot lines for the wall conditions
    plt.plot(xi_line, va_max_line, 'k:', label=r'$v = \xi$')
    plt.plot(xi_line, v_shock_line, 'k--', label=r'$v = v_{\rm sh}(\xi)$')
    plt.plot(xi_line, vb_max_line, 'k-.', label=r'$v = \mu(\xi, c_s)$')

    n_xi = deflag.shape[2]
    for i, xi0 in enumerate(xi0_array):
        deflag_v_b = deflag[0, i, :]
        deflag_w_b = deflag[1, i, :]
        deflag_xi_b = deflag[2, i, :]
        deflag_v_f = deflag[3, i, :]
        deflag_w_f = deflag[4, i, :]
        deflag_xi_f = deflag[5, i, :]

        # Grey out parts of line which are unphysical
        unphysical = np.logical_and(deflag_v_b - bubble.v_shock(deflag_xi_b) < 0,
                                    deflag_v_b - bubble.lorentz(deflag_xi_b, bubble.CS0) > 0)
        # But let's keep the unphysical points to look at
        deflag_v_b_grey = deflag_v_b[unphysical]
        deflag_xi_b_grey = deflag_xi_b[unphysical]
        deflag_v_b[unphysical] = np.nan
        # Unphysical doesn't quite work for last points, so ...
        deflag_v_b_grey[deflag_v_b_grey < 1e-4] = np.nan

        # Plot
        plt.plot(deflag_xi_f, deflag_v_f, color=[0.8, 0.8, 0.8])
        plt.plot(deflag_xi_b, deflag_v_b, 'k')
        plt.plot(deflag_xi_b_grey, deflag_v_b_grey, color=[0.8, 0.8, 0.8])

        # Make and plot a few lines starting from xi = 1
        if (i // 2) * 2 == i:
            det_v_b, det_w_b, det_xi_b, _ = bubble.fluid_integrate_param(xi0, 1, 1, t_end=tau_backwards_end, n_xi=n_xi)

    # Plot curves corresponding to selected solutions (c.f. Espinosa et al 2010)
    plot_v_excerpt(0.5, 0.263)
    plot_v_excerpt(0.7, 0.052)
    plot_v_excerpt(0.77, 0.091)

    # Make plot look nice
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$v(\xi)$')
    plt.grid()
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    # plt.savefig(spu.md_path + 'xi-v_plane_sols.pdf')
