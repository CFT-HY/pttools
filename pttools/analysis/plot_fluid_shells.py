r"""Plot $v(\xi)$ and $w(\xi)$ for fluid shells"""

import typing as tp

import matplotlib.pyplot as plt
import numpy as np

import pttools.type_hints as th
from pttools.analysis import utils
from pttools.bubble import boundary, check, const, fluid, props, quantities, relativity, shock, transition


def plot_fluid_shells(
        v_wall_list: th.FloatListOrArr,
        alpha_n_list: tp.Union[tp.List[float], np.ndarray],
        multi: bool = False,
        save_string: str = None,
        Np: int = const.N_XI_DEFAULT,
        debug: bool = False,
        draw: bool = None) -> tp.Union[plt.Figure, tp.Tuple[plt.Figure, np.ndarray]]:
    r"""
    Calls :func:`pttools.bubble.fluid.fluid_shell` and plots resulting v, w against xi.
    Annotates titles with:

    - Wall type, $v_\text{wall}, \alpha_n$
    - $\alpha_+$ ($\alpha$ just in front of wall)
    - $r$ (ratio of enthalpies either side of wall)
    - $\xi_{sh}$ (shock speed)
    - $\frac{w_0}{w_n}$ (ration of internal to external enthalpies)
    - ubar_f (mean square U = gamma(v) v)
    - K kinetic energy fraction
    - kappa (Espinosa et al efficiency factor)
    - omega (thermal energy relative to scalar potential energy, as measured by trace anomaly)

    Last two should sum to 1.

    :param v_wall_list: $v_\text{wall}$ values to simulate with
    :param alpha_n_list: $\alpha_n$ values to simulate with (same size as v_wall_list)
    :param multi: whether to plot multiple figures
    :param save_string: a descriptive name for the figure file. Required for saving the figure.
    :param Np: number of $\xi$ points
    :param debug: whether to return debug data
    :param draw: whether to actually fill the figure with data.
        This is used to disable drawing on systems that don't have LaTeX installed,
        such as the unit testing environment.
    :return: figure (or figure and data array if debug is True)
    """
    if draw is None:
        draw = utils.ENABLE_DRAWING
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

    fig: plt.Figure
    fig, ax = plt.subplots(2, ncols, figsize=(fig_width, 8), sharex='col', sharey='row', squeeze=False)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0.1)

    n = 0

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
        check.check_physical_params((v_wall, alpha_n))

        sol_type = transition.identify_solution_type_bag(v_wall, alpha_n)
        if sol_type == boundary.SolutionType.ERROR:
            raise RuntimeError(f"No solution for v_wall = {v_wall}, alpha_n = {alpha_n}.")

        v, w, xi = fluid.fluid_shell(v_wall, alpha_n, Np)
        n_cs = int(np.floor(const.CS0 * Np))
        n_sh = xi.size - 2
        v_sh = shock.v_shock_bag(xi_even)
        w_sh = shock.wm_shock_bag(xi_even)

        # n_wall = find_v_index(xi, v_wall)
        # n_cs = np.int(np.floor(cs0*Np))
        # n_sh = xi.size-2
        #
        # r = w[n_wall]/w[n_wall-1]
        # alpha_plus = alpha_n*w[-1]/w[n_wall]

        # Plot
        yscale_v = max(max(v), yscale_v)
        yscale_enth_max = max(max(w), yscale_enth_max)
        # yscale_enth_min = min(min(w),yscale_enth_min)
        yscale_enth_min = 2 * w[-1] - yscale_enth_max
        wn_max = max(w[-1], wn_max)

        # First velocity v
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

        # Then enthalpy w
        # ax[1,n].plot(xi, np.ones_like(xi)*w[-1], '--', color='0.5')
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
            # ax[1,n].set_title(r'$w_0/w_n = {:4.2}$, $\bar{{U}}_f = {:.3f}$, '
            # r'$K = {:5.3g}$, $\kappa = {:5.3f}$, $\omega = {:5.3f}$'.format(
            #     w[0]/w[-1],ubarf2**0.5,ke_frac, kappa, dw),size=14)
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
        fig.canvas.draw()
    ylabels = [tick.get_text() for tick in ax[1, 0].get_yticklabels()]
    # TODO: Fix the warning "FixedFormatter should only be used together with FixedLocator"
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
        return fig, np.array(data, dtype=np.float_)
    return fig
