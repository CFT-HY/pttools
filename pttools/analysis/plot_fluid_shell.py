r"""Plot $v(\xi)$ and $w(\xi)$ for a fluid shell"""

import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis import utils
from pttools.bubble import boundary, const, fluid_bag, relativity


def plot_fluid_shell(
        v_wall: float,
        alpha_n: float,
        save_string: str = None,
        Np: int = const.N_XI_DEFAULT,
        low_v_approx: bool = False,
        high_v_approx: bool = False,
        draw: bool = None) \
        -> tp.Tuple[plt.Figure, tp.Dict[str, tp.Union[np.ndarray, float]]]:
    r"""
    Calls :func:`pttools.bubble.fluid.fluid_shell` and plots resulting $v, w$ against $\xi$.
    Also plots:

    - shock curves (where $v$ and $w$ should form shock)
    - low alpha approximation if $\alpha_+ < 0.025$
    - high alpha approximation if $\alpha_+ > 0.2$

    Annotates titles with:

    - Wall type, $v_\text{wall}$, $\alpha_n$
    - $\alpha_+$ ($\alpha$ just in front of wall)
    - $r$ (ratio of enthalpies either side of wall)
    - $\xi_{sh}$ (shock speed)
    - $\frac{w_0}{w_n}$ (ration of internal to external enthalpies)
    - ubar_f (mean square $U = \gamma (v) v$)
    - K kinetic energy fraction
    - kappa (Espinosa et al. efficiency factor)
    - omega (thermal energy relative to scalar potential energy, as measured by trace anomaly)

    Last two should sum to 1.

    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param draw: whether to actually fill the figure with data.
        This is used to disable drawing on systems that don't have LaTeX installed,
        such as the unit testing environment.
    :return: figure, dict of generated params
    """
    if draw is None:
        draw = utils.ENABLE_DRAWING

    params = fluid_bag.fluid_shell_params(
        v_wall=v_wall,
        alpha_n=alpha_n,
        Np=Np,
        low_v_approx=low_v_approx,
        high_v_approx=high_v_approx
    )
    alpha_plus = params["alpha_plus"]
    dw = params["dw"]
    kappa = params["kappa"]
    ke_frac = params["ke_frac"]
    n_cs = params["n_cs"]
    n_sh = params["n_sh"]
    n_wall = params["n_wall"]
    r = params["r"]
    sol_type = params["sol_type"]
    ubarf2 = params["ubarf2"]
    v = params["v"]
    v_approx = params["v_approx"]
    v_sh = params["v_sh"]
    w = params["w"]
    w_approx = params["w_approx"]
    w_sh = params["w_sh"]
    xi = params["xi"]
    xi_even = params["xi_even"]

    # high_v_plot = 0.8 # Above which plot v ~ xi approximation
    # low_v_plot = 0.2  # Below which plot  low v approximation

    # Plot
    yscale_v = max(v) * 1.2
    xscale_max = min(xi[-2] * 1.1, 1.0)
    yscale_enth_max = max(w) * 1.2
    yscale_enth_min = min(w) / 1.2
    xscale_min = xi[n_wall] * 0.5

    fig = plt.figure(figsize=(7, 8))

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

    return fig, params
