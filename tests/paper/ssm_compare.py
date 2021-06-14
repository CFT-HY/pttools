"""Compare SSM prediction with data
Creates and plots velocity and GW power spectra from SSM

Modified from
https://bitbucket.org/hindmars/sound-shell-model/
"""

import enum
import logging
import os
import typing as tp

import numpy as np
import matplotlib.pyplot as plt

import pttools.bubble as b
import pttools.ssmtools as ssm
from tests.paper import const
from tests.paper import utils
from tests.test_utils import TEST_DATA_PATH

logger = logging.getLogger(__name__)

utils.setup_plotting()

MDP = os.path.join(TEST_DATA_PATH, "model_data/")
GDP = os.path.join(TEST_DATA_PATH, "graphs/")
if not os.path.isdir(MDP):
    os.mkdir(MDP)
if not os.path.isdir(GDP):
    os.mkdir(GDP)

# All run parameters

VW_INTER_LIST = [0.92, 0.72, 0.44]
# NB prace runs did not include intermediate 0.80, 0.56

T_WEAK_LIST = [1210, 1380, 1630, 1860, 2520]
T_INTER_LIST = [1180, 1480, 2650]

DT_WEAK_LIST = [0.08 * t for t in T_WEAK_LIST]
DT_INTER_LIST = [0.08 * t for t in T_INTER_LIST]
DT_INTER_LIST[1] = 0.05 * T_INTER_LIST[1]  # dx=1

STEP_WEAK_LIST = [int(round(dt / 20) * 20) for dt in DT_WEAK_LIST]
STEP_INTER_LIST = [int(round(dt / 20) * 20) for dt in DT_INTER_LIST]

# Files used in prace paper 2017
DIR_WEAK_LIST = [
    "results-weak-scaled_etatilde0.19_v0.92_dx2/",
    "results-weak-scaled_etatilde0.35_v0.80_dx2/",
    "results-weak-scaled_etatilde0.51_v0.68_dx2/",
    "results-weak-scaled_etatilde0.59_v0.56_dx2/",
    "results-weak-scaled_etatilde0.93_v0.44_dx2/"
]
DIR_INTER_LIST = [
    "results-intermediate-scaled_etatilde0.17_v0.92_dx2/",
    "results-intermediate-scaled_etatilde0.40_v0.72_dx1/",
    "results-intermediate-scaled_etatilde0.62_v0.44_dx2/"
]

PATH_HEAD = os.path.join(TEST_DATA_PATH, "fluidprofiles/")
PATH_LIST_ALL = ['weak/', 'intermediate/']
FILE_PATTERN = 'data-extracted.{:05d}.txt'

VW_LIST_ALL = [const.VW_WEAK_LIST, VW_INTER_LIST]
STEP_LIST_ALL = [STEP_WEAK_LIST, STEP_INTER_LIST]
DIR_LIST_ALL = [DIR_WEAK_LIST, DIR_INTER_LIST]


@enum.unique
class Method(str, enum.Enum):
    E_CONSERVING = "e_conserving"


@enum.unique
class PSType(str, enum.Enum):
    GW = "gw"
    V = "v"


@enum.unique
class Position(str, enum.Enum):
    HIGH = "high"
    LOW = "low"
    MED = "med"


@enum.unique
class Strength(str, enum.Enum):
    INTER = "inter"
    STRONG = "strong"
    WEAK = "weak"


def plot_ps(
        z_list,
        pow_list,
        ps_type: PSType,
        ax_limits: Strength = Strength.WEAK,
        leg_list=None,
        col_list=None,
        ls_list=None,
        fig: plt.Figure = None) -> plt.Figure:
    # Plots a list of power spectra, with axis limits appropriate to prace runs
    # returns a figure handle

    if col_list is None:
        col_list = ['b']*len(z_list)

    if ls_list is None:
        ls_list = ['-']*len(z_list)

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
            powers.append(np.trapz(power/z, z))
    else:
        for z, power, leg, col, ls in zip(z_list, pow_list, leg_list, col_list, ls_list):
            ax.loglog(z, power, color=col, linestyle=ls, label=leg)
            powers.append(np.trapz(power/z, z))

    # Pretty graphs
    ax.grid(True)
    ax.set_xlabel(r'$kR_*$')
    if ps_type is PSType.GW:
        ax.set_ylabel(r'$(H_{\rm n}R_*)^{-1}\mathcal{P}^\prime_{\rm ' + ps_type + '}(kR_*)$')
    else:
        ax.set_ylabel(r'$\mathcal{P}_{\rm ' + ps_type + '}(kR_*)$')
    ax.set_ylim([p_min, p_max])
    ax.set_xlim([const.Z_MIN, const.Z_MAX])
    if leg_list is not None:
        plt.legend(loc='best')
    plt.tight_layout()
    
    return fig


def generate_ps(
        vw: float,
        alpha: float,
        method: Method = Method.E_CONSERVING,
        v_xi_file=None,
        save_ids: tp.Tuple[str, str] = (None, None),
        show: bool = True,
        debug: bool = False):
    """ Generates, plots velocity and GW power as functions of $kR_*$.
    Saves power spectra in files pow_v_*, pow_gw_*...<string>.txt if save_id[0]=string.
    Saves plots in files pow_v_*, pow_gw_*...<string>.pdf if save_id[1]=string.
    Shows plots if show=True
    Returns <V^2> and Omgw divided by (Ht.HR*).
    """

    Np = const.NP_LIST[-1]
    col = const.COLOURS[0]

    if alpha < 0.01:
        strength = Strength.WEAK
    elif alpha < 0.1:  # intermediate transition
        strength = Strength.INTER
    else:
        logger.warning("alpha > 0.1, taking strength = strong")
        strength = Strength.STRONG

    # Generate power spectra
    z = np.logspace(np.log10(const.Z_MIN), np.log10(const.Z_MAX), Np[0])

    # Array using the minimum & maximum values set earlier, with Np[0] number of points
    print("vw = ", vw, "alpha = ", alpha, "Np = ", Np)

    sd_v = ssm.spec_den_v(z, [vw, alpha, const.NUC_TYPE, const.NUC_ARGS], Np[1:], method=method)
    pow_v = ssm.pow_spec(z,sd_v)
    V2_pow_v = np.trapz(pow_v/z, z)

    if v_xi_file is not None:
        sd_v2 = ssm.spec_den_v(z, [vw, alpha, const.NUC_TYPE, const.NUC_ARGS], Np[1:], v_xi_file, method=method)
        pow_v2 = ssm.pow_spec(z, sd_v2)
        V2_pow_v = np.trapz(pow_v2/z, z)

    sd_gw, y = ssm.spec_den_gw_scaled(z, sd_v)
    pow_gw = ssm.pow_spec(y, sd_gw)
    gw_power = np.trapz(pow_gw/y, y)    

    if v_xi_file is not None:
        sd_gw2, y = ssm.spec_den_gw_scaled(z, sd_v2)
        pow_gw2 = ssm.pow_spec(y, sd_gw2)
        gw_power = np.trapz(pow_gw2/y, y)    

    # Now for some plotting if requested
    if save_ids[1] is not None or show:
        f1 = plt.figure(figsize=[8, 4])
        ax_v = plt.gca()
        
        f2 = plt.figure(figsize=[8, 4])
        ax_gw = plt.gca()
        
        ax_gw.loglog(y, pow_gw, color=col)
        ax_v.loglog(z, pow_v, color=col)
        
        if v_xi_file is not None:    
            ax_v.loglog(z, pow_v2, color=col, linestyle='--')
            ax_gw.loglog(y, pow_gw2, color=col, linestyle='--')

        inter_flag = (abs(b.CS0 - vw) < 0.05)  # Due intermediate power law
        plot_guide_power_laws_prace(f1, f2, z, pow_v, y, pow_gw, inter_flag=inter_flag)

        # Pretty graph 1
        pv_min, pv_max = get_yaxis_limits(PSType.V, strength)
        ax_v.grid(True)
        ax_v.set_xlabel(r'$kR_*$')
        ax_v.set_ylabel(r'$\mathcal{P}_{\rm v}(kR_*)$')
        ax_v.set_ylim([pv_min, pv_max])
        ax_v.set_xlim([const.Z_MIN, const.Z_MAX])
        f1.tight_layout()

        # Pretty graph 2
        pgw_min, pgw_max = get_yaxis_limits(PSType.GW, strength)
        ax_gw.grid(True)
        ax_gw.set_xlabel(r'$kR_*$')
        ax_gw.set_ylabel(r'$\Omega^\prime_{\rm gw}(kR_*)/(H_{\rm n}R_*)$')
        ax_gw.set_ylim([pgw_min, pgw_max])
        ax_gw.set_xlim([const.Z_MIN, const.Z_MAX])
        f2.tight_layout()

    # Now some saving if requested
    if save_ids[0] is not None or save_ids[1] is not None:
        nz_string = 'nz{}k'.format(Np[0] // 1000)
        nx_string = '_nx{}k'.format(Np[1] // 1000)
        nT_string = '_nT{}-'.format(Np[2])
        file_suffix = "vw{:3.2f}alpha{}_".format(vw, alpha) + const.NUC_STRING + nz_string + nx_string + nT_string

    if save_ids[0] is not None:
        data_file_suffix = file_suffix + save_ids[0] + '.txt'

        if v_xi_file is None:
            np.savetxt(MDP + 'pow_v_' + data_file_suffix,
                       np.stack((z, pow_v), axis=-1), fmt='%.18e %.18e')
            np.savetxt(MDP + 'pow_gw_' + data_file_suffix,
                       np.stack((y, pow_gw), axis=-1), fmt='%.18e %.18e')
        else:
            np.savetxt(MDP + 'pow_v_' + data_file_suffix,
                       np.stack((z, pow_v, pow_v2), axis=-1), fmt='%.18e %.18e %.18e')
            np.savetxt(MDP + 'pow_gw_' + data_file_suffix,
                       np.stack((y, pow_gw, pow_gw2), axis=-1), fmt='%.18e %.18e %.18e')

    if save_ids[1] is not None:
        graph_file_suffix = file_suffix + save_ids[1] + '.pdf'
        f1.savefig(GDP + "pow_v_" + graph_file_suffix)
        f2.savefig(GDP + "pow_gw_" + graph_file_suffix)

    # Now some diagnostic comparisons between real space <v^2> and Fourier space already calculated
    v_ip, w_ip, xi = b.fluid_shell(vw, alpha)
    Ubarf2 = b.ubarf_squared(v_ip, w_ip, xi, vw)

    print("vw = {}, alpha = {}, nucleation = {}".format(vw, alpha, const.NUC_STRING))
    print("<v^2> =                      ", V2_pow_v)
    print("Ubarf2 (1 bubble)            ", Ubarf2)
    print("Ratio <v^2>/Ubarf2           ", V2_pow_v/Ubarf2)
    print("gw power (scaled):           ", gw_power)

    if show:
        plt.show()

    if debug:
        debug_data = [Ubarf2]
        debug_arrs = [sd_v, pow_v, gw_power]
        if v_xi_file is not None:
            debug_arrs += [sd_v2, pow_v2, sd_gw2, pow_gw2]
        debug_data += [np.sum(arr) for arr in debug_arrs]
        return V2_pow_v, gw_power, debug_data
    return V2_pow_v, gw_power

    
def all_generate_ps_prace(save_ids: tp.Tuple[str, str] = ('', ''), show=True, debug: bool = False):
    """Generate power spectra with Prace17 SSM parameters. 
    Save data files and graphs.
    Returns U-bar-f^2 and GW power as tuple of lists."""

    method = Method.E_CONSERVING

    v2_list = []
    Omgw_scaled_list = []
    debug_data = []

    for vw_list, alpha, step_list, path, dir_list in \
            zip(VW_LIST_ALL, const.ALPHA_LIST_ALL, STEP_LIST_ALL, PATH_LIST_ALL, DIR_LIST_ALL):
        for vw, step, dir_name in zip(vw_list, step_list, dir_list):
            v_xi_file = PATH_HEAD + path + dir_name + FILE_PATTERN.format(step)
            print("v_xi_file:", v_xi_file)
            if debug:
                v2, Omgw, data = generate_ps(vw, alpha, method, v_xi_file, save_ids, show, debug=debug)
                debug_data.append(data)
            else:
                v2, Omgw = generate_ps(vw, alpha, method, v_xi_file, save_ids, show)
            v2_list.append(v2)
            Omgw_scaled_list.append(Omgw)

    if debug:
        return v2_list, Omgw_scaled_list, debug_data
    return v2_list, Omgw_scaled_list


def get_yaxis_limits(ps_type: PSType, strength: Strength = Strength.WEAK):
    if strength is Strength.WEAK:
        if ps_type is PSType.V:
            p_min = 1e-8
            p_max = 1e-3
        elif ps_type is PSType.GW:
            p_min = 1e-16
            p_max = 1e-8
        else:
            p_min = 1e-8
            p_max = 1e-3
    elif strength is Strength.INTER:
        if ps_type is PSType.V:
            p_min = 1e-7
            p_max = 1e-2
        elif ps_type is PSType.GW:
            p_min = 1e-12
            p_max = 1e-4
        else:
            p_min = 1e-7
            p_max = 1e-2
    elif strength is Strength.STRONG:
        if ps_type is PSType.V:
            p_min = 1e-5
            p_max = 1
        elif ps_type is PSType.GW:
            p_min = 1e-8
            p_max = 1
        else:
            p_min = 1e-5
            p_max = 1e-1
    else:
        logger.warning("strength = [ *weak | inter | strong]")
        if ps_type is PSType.V:
            p_min = 1e-8
            p_max = 1e-3
        elif ps_type is PSType.GW:
            p_min = 1e-16
            p_max = 1e-8
        else:
            p_min = 1e-8
            p_max = 1e-3
    
    return p_min, p_max


# Functions for plotting guide power laws on graphs

def get_ymax_location(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Returns x, y coordinates of maximum of array y
    ymax = max(y)
    xmax = x[np.where(y == ymax)][0]
    return np.array([xmax, ymax])


def plot_guide_power_law_prace(ax: plt.Axes, x: np.ndarray, y: np.ndarray, n, position: Position, shifts=None):
    # Wrapper for plot_guide_power_law, with power laws and line 
    # shifts appropriate for velocity and GW spectra of prace runs
    if shifts is None:
        if position is Position.HIGH:
            line_shift = [2, 1]
            txt_shift = [1.25, 1]
            xloglen = 1
        elif position is Position.MED:
            line_shift = [0.5, 1.2]
            txt_shift = [0.5, 0.7]
            xloglen = -0.8
        elif position is Position.LOW:
            line_shift = [0.25, 0.25]
            txt_shift = [0.5, 0.5]
            xloglen = -1
        else:
            raise ValueError("Position not recognised")
    else:
        line_shift = shifts[0]
        txt_shift = shifts[1]
        if position is Position.HIGH:
            xloglen = 1
        elif position is Position.MED:
            xloglen = -0.8
        elif position is Position.LOW:
            xloglen = -1
        else:
            raise ValueError(f"Position not recognised: {position}")

    max_loc = get_ymax_location(x, y)

    power_law_loc = max_loc * np.array(line_shift)
    plot_guide_power_law(ax, power_law_loc, n, xloglen=xloglen, 
                         txt=f"$k^{{{n:}}}$", txt_shift=txt_shift)

    return power_law_loc


def plot_guide_power_law(
        ax: plt.Axes,
        loc: np.ndarray,
        power,
        xloglen=1,
        txt: str = "",
        txt_shift: tp.Tuple[float, float] = (1, 1),
        color: str = "k",
        linestyle: str = "-"):
    # Plot a guide power law going through loc[0], loc[1] with index power
    # Optional annotation at (loc[0]*txt_shift[0], loc[1]*txt_shift[1])
    # Returns the points in two arrays (is this the best thing?)
    xp = loc[0]
    yp = loc[1]
    x_guide = np.logspace(np.log10(xp), np.log10(xp) + xloglen, 2)
    y_guide = yp*(x_guide/xp)**power
    ax.loglog(x_guide, y_guide, color=color, linestyle=linestyle)

    if txt:
        txt_loc = loc * np.array(txt_shift)
        ax.text(txt_loc[0], txt_loc[1], txt, fontsize=16)

    return x_guide, y_guide


def plot_guide_power_laws_ssm(
        f: plt.Figure,
        z: np.ndarray,
        powers: np.ndarray,
        ps_type: PSType = PSType.V,
        inter_flag: bool = False) -> plt.Figure:
    # Plot guide power laws (assumes params all same for list)
    # Shifts designed for simultaneous nucleation lines
    x_high = 10
    x_low = 3
    if ps_type is PSType.V:
        n_lo, n_med, n_hi = 5, 1, -1
        shifts_hi = [[2, 1], [1.5, 1]]
        shifts_lo = [[0.5, 0.25], [0.5, 0.15]]
    elif ps_type is PSType.GW:
        n_lo, n_med, n_hi = 9, 1, -3
        shifts_hi = None
        shifts_lo = [[0.6, 0.25], [0.5, 0.08]]
    else:
        # n_lo, n_med, n_hi = tuple(ps_type)
        raise NotImplementedError("TODO: define shifts_hi and shifts_lo")

    logger.debug("Plotting guide power laws")

    high_peak = np.where(z > x_high)
    plot_guide_power_law_prace(f.axes[0], z[high_peak], powers[high_peak], n_hi, Position.HIGH, shifts=shifts_hi)

    if inter_flag:
        # intermediate power law to be plotted
        plot_guide_power_law_prace(f.axes[0], z[high_peak], powers[high_peak], n_med, Position.MED)

    low_peak = np.where(z < x_low)
    plot_guide_power_law_prace(f.axes[0], z[low_peak], powers[low_peak], n_lo, Position.LOW, shifts=shifts_lo)

    return f


def plot_guide_power_laws_prace(
        f_v: plt.Figure,
        f_gw: plt.Figure,
        z: np.ndarray,
        pow_v: np.ndarray,
        y: np.ndarray,
        pow_gw: np.ndarray,
        np_lo: tp.Tuple[int, int] = (5, 9),
        inter_flag: bool = False) -> tp.Tuple[plt.Figure, plt.Figure]:
    # Plot guide power laws (assumes params all same for list)
    # Shifts designed for simulataneous nucleation lines
    x_high = 10
    x_low = 2
    [nv_lo, ngw_lo] = np_lo
    logger.debug("Plotting guide power laws")
    high_peak_v = np.where(z > x_high)
    high_peak_gw = np.where(y > x_high)
    plot_guide_power_law_prace(
        f_v.axes[0], z[high_peak_v], pow_v[high_peak_v], -1, Position.HIGH,
        shifts=[[2, 1], [2.1, 0.6]])
    plot_guide_power_law_prace(f_gw.axes[0], y[high_peak_gw], pow_gw[high_peak_gw], -3, Position.HIGH)

    if inter_flag:
        # intermediate power law to be plotted
        plot_guide_power_law_prace(f_v.axes[0], z[high_peak_v], pow_v[high_peak_v], 1, Position.MED)
        plot_guide_power_law_prace(
            f_gw.axes[0], y[high_peak_gw], pow_gw[high_peak_gw], 1, Position.MED,
            shifts=[[0.5, 1.5], [0.5, 2]])

    low_peak_v = np.where(z < x_low)
    low_peak_gw = np.where(y < x_low)
    plot_guide_power_law_prace(
        f_v.axes[0], z[low_peak_v], pow_v[low_peak_v], nv_lo, Position.LOW,
        shifts=[[0.5, 0.25], [0.5, 0.15]])
#    plot_guide_power_law_prace(f_gw.axes[0], y[low_peak_gw], pow_gw[low_peak_gw], ngw_lo, 'low',
#                               shifts=[[0.4,0.5],[0.5,0.5]])
    plot_guide_power_law_prace(
        f_gw.axes[0], y[low_peak_gw], pow_gw[low_peak_gw], ngw_lo, Position.LOW,
        shifts=[[0.6, 0.25], [0.5, 0.08]])

    return f_v, f_gw
