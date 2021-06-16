"""Plotting functions of the bubble module

All functions using Matplotlib etc. should be put here to keep them separate from the numerical codebase.
"""

import typing as tp

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import approx
from . import boundary
from . import check
from . import const
from . import fluid
from . import props
from . import quantities
from . import relativity
from . import transition

FLOAT_LIST_OR_ARR_TYPE = tp.Union[tp.List[float], np.ndarray]


def setup_plotting(font: str = "serif", font_size: int = 20, usetex: bool = True):
    """Get decent-sized plots

    LaTeX can cause problems if system not configured correctly
    """
    plt.rc('text', usetex=usetex)
    plt.rc('font', family=font)

    mpl.rcParams.update({'font.size': font_size})
    mpl.rcParams.update({'lines.linewidth': 1.5})
    mpl.rcParams.update({'axes.linewidth': 2.0})
    mpl.rcParams.update({'axes.labelsize': font_size})
    mpl.rcParams.update({'xtick.labelsize': font_size})
    mpl.rcParams.update({'ytick.labelsize': font_size})
    # but make legend smaller
    mpl.rcParams.update({'legend.fontsize': 14})


# setup_plotting()


def plot_fluid_shell(
        v_wall: float,
        alpha_n: float,
        save_string: str = None,
        Np: int = const.N_XI_DEFAULT,
        low_v_approx: bool = False,
        high_v_approx: bool = False,
        debug: bool = False,
        draw: bool = True) \
        -> tp.Union[plt.Figure, tp.Tuple[plt.Figure, tp.List[np.ndarray], tp.List[tp.Union[int, float]]]]:
    """
    Calls ``fluid_shell`` and plots resulting v, w against xi, returning figure handle.
    Also plots:
    - shock curves (where v and w should form shock)
    - low alpha approximation if alpha_plus < 0.025
    - high alpha approximation if alpha_plus > 0.2
    Annotates titles with:
    - Wall type, v_wall, alpha_n
    - alpha_plus (alpha just in front of wall)
    - r (ratio of enthalpies either side of wall)
    - xi_sh (shock speed)
    - w_0/w_n (ration of internal to external enthalpies)
    - ubar_f (mean square U = gamma(v) v)
    - K kinetic energy fraction
    - kappa (Espinosa et al efficiency factor)
    - omega (thermal energy relative to scalar potential energy, as measured by trace anomaly)
    Last two should sum to 1.
    """
    check.check_physical_params([v_wall, alpha_n])

    #    high_v_plot = 0.8 # Above which plot v ~ xi approximation
    #    low_v_plot = 0.2  # Below which plot  low v approximation

    sol_type = transition.identify_solution_type(v_wall, alpha_n)

    if sol_type is boundary.SolutionType.ERROR:
        raise RuntimeError(f"No solution for v_wall = {v_wall}, alpha_n = {alpha_n}")

    v, w, xi = fluid.fluid_shell(v_wall, alpha_n, Np)

    # vmax = max(v)

    xi_even = np.linspace(1 / Np, 1 - 1 / Np, Np)
    v_sh = props.v_shock(xi_even)
    w_sh = props.w_shock(xi_even)

    n_wall = props.find_v_index(xi, v_wall)
    n_cs = int(np.floor(const.CS0 * Np))
    n_sh = xi.size - 2

    r = w[n_wall] / w[n_wall - 1]
    alpha_plus = alpha_n * w[-1] / w[n_wall]

    ubarf2 = quantities.ubarf_squared(v, w, xi, v_wall)
    # Kinetic energy fraction of total (Bag equation of state)
    ke_frac = ubarf2 / (0.75 * (1 + alpha_n))
    # Efficiency of turning Higgs potential into kinetic energy
    kappa = ubarf2 / (0.75 * alpha_n)
    # and efficiency of turning Higgs potential into thermal energy
    dw = 0.75 * quantities.mean_enthalpy_change(v, w, xi, v_wall) / (0.75 * alpha_n * w[-1])

    if high_v_approx:
        v_approx = approx.v_approx_high_alpha(xi[n_wall:n_sh], v_wall, v[n_wall])
        w_approx = approx.w_approx_high_alpha(xi[n_wall:n_sh], v_wall, v[n_wall], w[n_wall])

    if low_v_approx:
        v_approx = approx.v_approx_low_alpha(xi, v_wall, alpha_plus)
        w_approx = approx.w_approx_low_alpha(xi, v_wall, alpha_plus)
    # Plot
    yscale_v = max(v) * 1.2
    xscale_max = min(xi[-2] * 1.1, 1.0)
    yscale_enth_max = max(w) * 1.2
    yscale_enth_min = min(w) / 1.2
    xscale_min = xi[n_wall] * 0.5

    f = plt.figure(figsize=(7, 8))

    # First velocity
    plt.subplot(2, 1, 1)

    plt.title(
        rf'$\xi_{{\rm w}} =  {v_wall}$, $\alpha_{{\rm n}} =  {alpha_n:.3}$, '
        rf'$\alpha_+ =  {alpha_plus:5.3f}$, $r =  {r:.3f}$, $\xi_{{\rm sh}} =  {xi[-2]:5.3f}$', size=16)
    plt.plot(xi, v, 'b', label=r'$v(\xi)$')

    if not sol_type == boundary.SolutionType.DETON:
        plt.plot(xi_even[n_cs:], v_sh[n_cs:], 'k--', label=r'$v_{\rm sh}(\xi_{\rm sh})$')
        if high_v_approx:
            plt.plot(xi[n_wall:n_sh], v_approx, 'b--', label=r'$v$ ($v < \xi$ approx)')
            plt.plot(xi, xi, 'k--', label=r'$v = \xi$')

    if not sol_type == boundary.SolutionType.SUB_DEF:
        v_minus_max = relativity.lorentz(xi_even, const.CS0)
        plt.plot(xi_even[n_cs:], v_minus_max[n_cs:], 'k-.', label=r'$\mu(\xi,c_{\rm s})$')

    if low_v_approx:
        plt.plot(xi, v_approx, 'b--', label=r'$v$ low $\alpha$ approx')

    plt.legend(loc='best')

    plt.ylabel(r'$v(\xi)$')
    plt.xlabel(r'$\xi$')
    plt.axis([xscale_min, xscale_max, 0.0, yscale_v])
    plt.grid()

    # Then enthalpy
    plt.subplot(2, 1, 2)

    plt.title(
        rf'$w_0/w_n = {w[0] / w[-1]:4.2}$, $\bar{{U}}_f = {ubarf2 ** 0.5:.3f}$, $K = {ke_frac:5.3g}$, '
        rf'$\kappa = {kappa:5.3f}$, $\omega = {dw:5.3f}$', size=16)
    plt.plot(xi, np.ones_like(xi) * w[-1], '--', color='0.5')
    plt.plot(xi, w, 'b', label=r'$w(\xi)$')

    if not sol_type == boundary.SolutionType.DETON:
        plt.plot(xi_even[n_cs:], w_sh[n_cs:], 'k--', label=r'$w_{\rm sh}(\xi_{\rm sh})$')

        if high_v_approx:
            plt.plot(xi[n_wall:n_sh], w_approx[:], 'b--', label=r'$w$ ($v < \xi$ approx)')

    else:
        wmax_det = (xi_even / const.CS0) * relativity.gamma2(xi_even) / relativity.gamma2(const.CS0)
        plt.plot(xi_even[n_cs:], wmax_det[n_cs:], 'k-.', label=r'$w_{\rm max}$')

    if low_v_approx:
        plt.plot(xi, w_approx, 'b--', label=r'$w$ low $\alpha$ approx')

    plt.legend(loc='best')
    plt.ylabel(r'$w(\xi)$', size=16)
    plt.xlabel(r'$\xi$', size=16)
    plt.axis([xscale_min, xscale_max, yscale_enth_min, yscale_enth_max])
    plt.grid()

    if draw:
        plt.tight_layout()

    if save_string is not None:
        plt.savefig(f"shell_plot_vw_{v_wall}_alphan_{alpha_n:.3}{save_string}")

    if debug:
        arrs = [v, w, xi, v_sh, w_sh]
        scalars = [n_wall, n_cs, n_sh, r, alpha_plus, ubarf2, ke_frac, kappa, dw]
        return f, arrs, scalars
    return f


def plot_fluid_shells(
        v_wall_list: FLOAT_LIST_OR_ARR_TYPE,
        alpha_n_list: tp.Union[tp.List[float], np.ndarray],
        multi: bool = False,
        save_string: str = None,
        Np: int = const.N_XI_DEFAULT,
        debug: bool = False,
        draw: bool = True) -> tp.Union[plt.Figure, tp.Tuple[plt.Figure, np.ndarray]]:
    """
    Calls ``fluid_shell`` and plots resulting v, w against xi. Returns figure handle.
    Annotates titles with:
    - Wall type, v_wall, alpha_n
    - alpha_plus (alpha just in front of wall)
    - r (ratio of enthalpies either side of wall)
    - xi_sh (shock speed)
    - w_0/w_n (ration of internal to external enthalpies)
    - ubar_f (mean square U = gamma(v) v)
    - K kinetic energy fraction
    - kappa (Espinosa et al efficiency factor)
    - omega (thermal energy relative to scalar potential energy, as measured by trace anomaly)
    Last two should sum to 1.
    """

    xi_even = np.linspace(1 / Np, 1 - 1 / Np, Np)
    yscale_v = 0.0
    yscale_enth_max = 1.0
    yscale_enth_min = 1.0
    wn_max = 0.0

    ncols = 1
    fig_width = 8
    if multi is True:
        ncols = len(v_wall_list)
        fig_width = ncols * 5

    f, ax = plt.subplots(2, ncols, figsize=(fig_width, 8), sharex='col', sharey='row', squeeze=False)
    f.subplots_adjust(hspace=0)
    f.subplots_adjust(wspace=0.1)

    n = 0

    if debug:
        # Debug arrays
        lst_v = []
        lst_w = []
        lst_xi = []
        lst_v_sh = []
        # lst_v_minus_max = []
        lst_w_sh = []

        # Debug values
        lst_ubarf2 = []
        lst_ke_frac = []
        lst_kappa = []
        lst_dw = []

    for v_wall, alpha_n in zip(v_wall_list, alpha_n_list):
        check.check_physical_params([v_wall, alpha_n])

        sol_type = transition.identify_solution_type(v_wall, alpha_n)

        if sol_type == boundary.SolutionType.ERROR:
            raise RuntimeError(f"No solution for v_wall = {v_wall}, alpha_n = {alpha_n}.")

        v, w, xi = fluid.fluid_shell(v_wall, alpha_n, Np)
        n_cs = int(np.floor(const.CS0 * Np))
        n_sh = xi.size - 2
        v_sh = props.v_shock(xi_even)
        w_sh = props.w_shock(xi_even)
        #
        #        n_wall = find_v_index(xi, v_wall)
        #        n_cs = np.int(np.floor(cs0*Np))
        #        n_sh = xi.size-2

        #        r = w[n_wall]/w[n_wall-1]
        #        alpha_plus = alpha_n*w[-1]/w[n_wall]
        #
        #
        # Plot
        yscale_v = max(max(v), yscale_v)
        yscale_enth_max = max(max(w), yscale_enth_max)
        #        yscale_enth_min = min(min(w),yscale_enth_min)
        yscale_enth_min = 2 * w[-1] - yscale_enth_max
        wn_max = max(w[-1], wn_max)

        # First velocity
        ax[0, n].plot(xi, v, 'b')
        if not sol_type == boundary.SolutionType.DETON:
            ax[0, n].plot(xi_even[n_cs:], v_sh[n_cs:], 'k--', label=r'$v_{\rm sh}(\xi_{\rm sh})$')
        if not sol_type == boundary.SolutionType.SUB_DEF:
            v_minus_max = relativity.lorentz(xi_even, const.CS0)
            ax[0, n].plot(xi_even[n_cs:], v_minus_max[n_cs:], 'k-.', label=r'$\mu(\xi,c_{\rm s})$')

        if multi:
            n_wall = props.find_v_index(xi, v_wall)
            r = w[n_wall] / w[n_wall - 1]
            alpha_plus = alpha_n * w[-1] / w[n_wall]

            ax[0, n].set_title(
                rf'$\alpha_{{\rm n}} =  {alpha_n:5.3f}$, $\alpha_+ =  {alpha_plus:5.3f}$, '
                rf'$r =  {r:5.3f}$, $\xi_{{\rm sh}} =  {xi[-2]:5.3f}$', size=14)

        ax[0, n].grid(True)

        # Then enthalpy
        #        ax[1,n].plot(xi, np.ones_like(xi)*w[-1], '--', color='0.5')
        ax[1, n].plot(xi, w, 'b')
        if not sol_type == boundary.SolutionType.DETON:
            ax[1, n].plot(xi_even[n_cs:n_sh], w_sh[n_cs:n_sh], 'k--', label=r'$w_{\rm sh}(\xi_{\rm sh})$')
        else:
            wmax_det = (xi_even / const.CS0) * relativity.gamma2(xi_even) / relativity.gamma2(const.CS0)
            ax[1, n].plot(xi_even[n_cs:], wmax_det[n_cs:], 'k-.', label=r'$w_{\rm max}$')

        if multi:
            ubarf2 = quantities.ubarf_squared(v, w, xi, v_wall)
            # Kinetic energy fraction of total (Bag equation of state)
            ke_frac = ubarf2 / (0.75 * (1 + alpha_n))
            # Efficiency of turning Higgs potential into kinetic energy
            kappa = ubarf2 / (0.75 * alpha_n)
            # and efficiency of turning Higgs potential into thermal energy
            dw = 0.75 * quantities.mean_enthalpy_change(v, w, xi, v_wall) / (0.75 * alpha_n * w[-1])
            #            ax[1,n].set_title(r'$w_0/w_n = {:4.2}$, $\bar{{U}}_f = {:.3f}$, '
            #            r'$K = {:5.3g}$, $\kappa = {:5.3f}$, $\omega = {:5.3f}$'.format(
            #                      w[0]/w[-1],ubarf2**0.5,ke_frac, kappa, dw),size=14)
            ax[1, n].set_title(rf'$K = {ke_frac:5.3g}$, $\kappa = {kappa:5.3f}$, $\omega = {dw:5.3f}$', size=14)

            if debug:
                # Scalars
                lst_ubarf2.append(ubarf2)
                lst_ke_frac.append(ke_frac)
                lst_kappa.append(kappa)
                lst_dw.append(dw)

        ax[1, n].set_xlabel(r'$\xi$')
        ax[1, n].grid(True)
        if multi:
            n += 1

        if debug:
            # Arrays
            lst_v.append(v)
            lst_w.append(w)
            lst_xi.append(xi)
            lst_v_sh.append(v_sh)
            lst_w_sh.append(w_sh)

    xscale_min = 0.0
    xscale_max = 1.0
    y_scale_enth_min = max(0.0, yscale_enth_min)
    ax[0, 0].axis([xscale_min, xscale_max, 0.0, yscale_v * 1.2])
    ax[1, 0].axis([xscale_min, xscale_max, y_scale_enth_min, yscale_enth_max * 1.1 - 0.1 * wn_max])

    if draw:
        f.canvas.draw()
    ylabels = [tick.get_text() for tick in ax[1, 0].get_yticklabels()]
    ax[1, 0].set_yticklabels(ylabels[:-1])

    ax[0, 0].set_ylabel(r'$v(\xi)$')
    if draw:
        plt.tight_layout()

    ax[1, 0].set_ylabel(r'$w(\xi)$')
    if draw:
        plt.tight_layout()

    if save_string is not None:
        plt.savefig(
            f"shells_plot_vw_{v_wall_list[0]}-{v_wall_list[-1]}_"
            f"alphan_{alpha_n_list[0]:.3}-{alpha_n_list[-1]:.3}{save_string}")

    if debug:
        data = [[np.nansum(arr) for arr in lst] for lst in [lst_v, lst_w, lst_xi, lst_v_sh, lst_w_sh]]
        if multi:
            data += [lst_ubarf2, lst_ke_frac, lst_kappa, lst_dw]
        return f, np.array(data, dtype=np.float_)
    return f
