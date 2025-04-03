"""Plotting methods for plotting guide power laws on graphs.

These methods are shared by
:mod:`tests.paper.ssm_paper_utils`
and
:mod:`tests.paper.ssm_compare`.
In the :ssm_repo:`sound-shell-model repository <>` these methods are included in both of the aforementioned files.
"""

import logging
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from tests.paper import const
from tests.paper import utils

logger = logging.getLogger(__name__)


def get_yaxis_limits(ps_type: utils.PSType, strength: utils.Strength = utils.Strength.WEAK) -> tp.Tuple[float, float]:
    if strength is utils.Strength.WEAK:
        if ps_type is utils.PSType.V:
            p_min = 1e-8
            p_max = 1e-3
        elif ps_type is utils.PSType.GW:
            p_min = 1e-16
            p_max = 1e-8
        else:
            p_min = 1e-8
            p_max = 1e-3
    elif strength is utils.Strength.INTER:
        if ps_type is utils.PSType.V:
            p_min = 1e-7
            p_max = 1e-2
        elif ps_type is utils.PSType.GW:
            p_min = 1e-12
            p_max = 1e-4
        else:
            p_min = 1e-7
            p_max = 1e-2
    elif strength is utils.Strength.STRONG:
        if ps_type is utils.PSType.V:
            p_min = 1e-5
            p_max = 1
        elif ps_type is utils.PSType.GW:
            p_min = 1e-8
            p_max = 1
        else:
            p_min = 1e-5
            p_max = 1e-1
    else:
        logger.warning("strength = [ *weak | inter | strong]")
        if ps_type is utils.PSType.V:
            p_min = 1e-8
            p_max = 1e-3
        elif ps_type is utils.PSType.GW:
            p_min = 1e-16
            p_max = 1e-8
        else:
            p_min = 1e-8
            p_max = 1e-3

    return p_min, p_max


def plot_guide_power_law(
        ax: plt.Axes,
        loc: np.ndarray,
        power,
        xloglen=1,
        txt: str = "",
        txt_shift: tp.Tuple[float, float] = (1, 1),
        color: str = "k",
        linestyle: str = "-"):
    """
    Plot a guide power law going through loc[0], loc[1] with index power
    Optional annotation at (loc[0]*txt_shift[0], loc[1]*txt_shift[1])
    Returns the points in two arrays (is this the best thing?)
    """
    xp = loc[0]
    yp = loc[1]
    x_guide = np.logspace(np.log10(xp), np.log10(xp) + xloglen, 2)
    y_guide = yp*(x_guide/xp)**power
    ax.loglog(x_guide, y_guide, color=color, linestyle=linestyle)

    if txt:
        txt_loc = loc * np.array(txt_shift)
        ax.text(txt_loc[0], txt_loc[1], txt, fontsize=16)

    return x_guide, y_guide


def plot_guide_power_law_prace(ax: plt.Axes, x: np.ndarray, y: np.ndarray, n, position: utils.Position, shifts=None):
    """
    Wrapper for plot_guide_power_law, with power laws and line
    shifts appropriate for velocity and GW spectra of prace runs
    """
    if shifts is None:
        if position is utils.Position.HIGH:
            line_shift = [2, 1]
            txt_shift = [1.25, 1]
            xloglen = 1
        elif position is utils.Position.MED:
            line_shift = [0.5, 1.2]
            txt_shift = [0.5, 0.7]
            xloglen = -0.8
        elif position is utils.Position.LOW:
            line_shift = [0.25, 0.25]
            txt_shift = [0.5, 0.5]
            xloglen = -1
        else:
            raise ValueError("Position not recognised")
    else:
        line_shift = shifts[0]
        txt_shift = shifts[1]
        if position is utils.Position.HIGH:
            xloglen = 1
        elif position is utils.Position.MED:
            xloglen = -0.8
        elif position is utils.Position.LOW:
            xloglen = -1
        else:
            raise ValueError(f"Position not recognised: {position}")

    max_loc = utils.get_ymax_location(x, y)

    power_law_loc = max_loc * np.array(line_shift)
    plot_guide_power_law(ax, power_law_loc, n, xloglen=xloglen,
                         txt=f"$k^{{{n:}}}$", txt_shift=txt_shift)

    return power_law_loc


def plot_guide_power_laws_prace(
        f_v: plt.Figure,
        f_gw: plt.Figure,
        z: np.ndarray,
        pow_v: np.ndarray,
        y: np.ndarray,
        pow_gw: np.ndarray,
        np_lo: tp.Tuple[int, int] = (5, 9),
        inter_flag: bool = False) -> tp.Tuple[plt.Figure, plt.Figure]:
    """
    Plot guide power laws (assumes params all same for list)
    Shifts designed for simulataneous nucleation lines
    """
    x_high = 10
    x_low = 2
    [nv_lo, ngw_lo] = np_lo
    logger.debug("Plotting guide power laws")
    high_peak_v = np.where(z > x_high)
    high_peak_gw = np.where(y > x_high)
    plot_guide_power_law_prace(
        f_v.axes[0], z[high_peak_v], pow_v[high_peak_v], -1, utils.Position.HIGH,
        shifts=[[2, 1], [2.1, 0.6]])
    plot_guide_power_law_prace(f_gw.axes[0], y[high_peak_gw], pow_gw[high_peak_gw], -3, utils.Position.HIGH)

    if inter_flag:
        # intermediate power law to be plotted
        plot_guide_power_law_prace(f_v.axes[0], z[high_peak_v], pow_v[high_peak_v], 1, utils.Position.MED)
        plot_guide_power_law_prace(
            f_gw.axes[0], y[high_peak_gw], pow_gw[high_peak_gw], 1, utils.Position.MED,
            shifts=[[0.5, 1.5], [0.5, 2]])

    low_peak_v = np.where(z < x_low)
    low_peak_gw = np.where(y < x_low)
    plot_guide_power_law_prace(
        f_v.axes[0], z[low_peak_v], pow_v[low_peak_v], nv_lo, utils.Position.LOW,
        shifts=[[0.5, 0.25], [0.5, 0.15]])
    # plot_guide_power_law_prace(f_gw.axes[0], y[low_peak_gw], pow_gw[low_peak_gw], ngw_lo, 'low',
    #                            shifts=[[0.4,0.5],[0.5,0.5]])
    plot_guide_power_law_prace(
        f_gw.axes[0], y[low_peak_gw], pow_gw[low_peak_gw], ngw_lo, utils.Position.LOW,
        shifts=[[0.6, 0.25], [0.5, 0.08]])

    return f_v, f_gw


def plot_guide_power_laws_ssm(
        fig: plt.Figure,
        z: np.ndarray,
        powers: np.ndarray,
        ps_type: utils.PSType = utils.PSType.V,
        inter_flag: bool = False) -> plt.Figure:
    """
    Plot guide power laws (assumes params all same for list)
    Shifts designed for simultaneous nucleation lines
    """
    x_high = 10
    x_low = 3
    if ps_type is utils.PSType.V:
        n_lo, n_med, n_hi = 5, 1, -1
        shifts_hi = [[2, 1], [1.5, 1]]
        shifts_lo = [[0.5, 0.25], [0.5, 0.15]]
    elif ps_type is utils.PSType.GW:
        n_lo, n_med, n_hi = 9, 1, -3
        shifts_hi = None
        shifts_lo = [[0.6, 0.25], [0.5, 0.08]]
    else:
        # n_lo, n_med, n_hi = tuple(ps_type)
        raise NotImplementedError("TODO: define shifts_hi and shifts_lo")

    logger.debug("Plotting guide power laws")

    high_peak = np.where(z > x_high)
    plot_guide_power_law_prace(fig.axes[0], z[high_peak], powers[high_peak], n_hi, utils.Position.HIGH, shifts=shifts_hi)

    if inter_flag:
        # intermediate power law to be plotted
        plot_guide_power_law_prace(fig.axes[0], z[high_peak], powers[high_peak], n_med, utils.Position.MED)

    low_peak = np.where(z < x_low)
    plot_guide_power_law_prace(fig.axes[0], z[low_peak], powers[low_peak], n_lo, utils.Position.LOW, shifts=shifts_lo)

    return fig


def plot_ps(
        z_list,
        pow_list,
        ps_type: utils.PSType,
        ax_limits: utils.Strength = utils.Strength.WEAK,
        leg_list=None,
        col_list=None,
        ls_list=None,
        fig: plt.Figure = None,
        pretty: bool = False) -> plt.Figure:
    """
    Plots a list of power spectra, with axis limits appropriate to prace runs
    returns a figure handle
    """
    if col_list is None:
        col_list = ['b'] * len(z_list)

    if ls_list is None:
        ls_list = ['-'] * len(z_list)

    if fig is None:
        fig = plt.figure(figsize=[8, 4])
        ax = plt.gca()
    else:
        raise NotImplementedError("TODO: axis is not defined")

    p_min, p_max = get_yaxis_limits(ps_type, ax_limits)

    powers = []

    if leg_list is None:
        for z, power, col, ls in zip(z_list, pow_list, col_list, ls_list):
            ax.loglog(z, power, color=col, linestyle=ls)
            powers.append(np.trapezoid(power / z, z))
    else:
        for z, power, leg, col, ls in zip(z_list, pow_list, leg_list, col_list, ls_list):
            ax.loglog(z, power, color=col, linestyle=ls, label=leg)
            powers.append(np.trapezoid(power / z, z))

    # Pretty graphs
    if pretty:
        ax.grid(True)
        ax.set_xlabel(r'$kR_*$')
        if ps_type is utils.PSType.GW:
            ax.set_ylabel(r'$(H_{\rm n}R_*)^{-1}\mathcal{P}^\prime_{\rm ' + ps_type + '}(kR_*)$')
        else:
            ax.set_ylabel(r'$\mathcal{P}_{\rm ' + ps_type + '}(kR_*)$')
        ax.set_ylim([p_min, p_max])
        ax.set_xlim([const.Z_MIN, const.Z_MAX])
        if leg_list is not None:
            plt.legend(loc='best')
        plt.tight_layout()

    return fig
