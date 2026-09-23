"""Compare SSM prediction with data
Creates and plots velocity and GW power spectra from SSM.

Modified from
:ssm_repo:`paper_ssm_prace/figures/ssm_compare.py`
"""

import logging
import os
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools import bubble, ssm
import pttools.type_hints as th
from tests.paper import const, plotting, utils
from tests.utils.const import TEST_DATA_PATH, TEST_FIGURE_PATH

logger: logging.Logger = logging.getLogger(__name__)

# bubble.setup_plotting()

MDP: str = os.path.join(TEST_DATA_PATH, "model_data")
GDP = TEST_FIGURE_PATH
# os.makedirs(MDP, exist_ok=True)
# os.makedirs(GDP, exist_ok=True)

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

PATH_HEAD: str = os.path.join(TEST_DATA_PATH, "fluidprofiles/")
PATH_LIST_ALL = ['weak/', 'intermediate/']
FILE_PATTERN = 'data-extracted.{:05d}.txt'

VW_LIST_ALL = [const.VW_WEAK_LIST, VW_INTER_LIST]
STEP_LIST_ALL = [STEP_WEAK_LIST, STEP_INTER_LIST]
DIR_LIST_ALL = [DIR_WEAK_LIST, DIR_INTER_LIST]


@tp.overload
def generate_ps(
        vw: float,
        alpha: float,
        method: ssm.Method = ssm.Method.E_CONSERVING,
        v_xi_file: str | None = None,
        save_ids: tuple[str | None, str | None] = (None, None),
        show: bool = True,
        debug: tp.Literal[False] = False) -> tuple[float, float]: ...


@tp.overload
def generate_ps(
        vw: float,
        alpha: float,
        method: ssm.Method = ssm.Method.E_CONSERVING,
        v_xi_file: str | None = None,
        save_ids: tuple[str | None, str | None] = (None, None),
        show: bool = True,
        *,
        debug: tp.Literal[True]) -> tuple[float, float, list[float]]: ...


def generate_ps(
        vw: float,
        alpha: float,
        method: ssm.Method = ssm.Method.E_CONSERVING,
        v_xi_file: str | None = None,
        save_ids: tuple[str | None, str | None] = (None, None),
        show: bool = True,
        debug: bool = False) -> tuple[float, float] | tuple[float, float, list[float]]:
    """
    Generates, plots velocity and GW power as functions of $kR_*$.
    Saves power spectra in files pow_v_*, pow_gw_*...<string>.txt if save_id[0]=string.
    Saves plots in files pow_v_*, pow_gw_*...<string>.pdf if save_id[1]=string.
    Shows plots if show=True
    Returns <V^2> and Omgw divided by (Ht.HR*).
    """
    Np = const.NP_ARR[-1]
    col = const.COLOURS[0]

    if alpha < 0.01:
        strength = utils.Strength.WEAK
    elif alpha < 0.1:  # intermediate transition
        strength = utils.Strength.INTER
    else:
        logger.warning("alpha > 0.1, taking strength = strong")
        strength = utils.Strength.STRONG

    # Generate power spectra
    z = np.logspace(np.log10(const.Z_MIN), np.log10(const.Z_MAX), Np[0])

    # Array using the minimum & maximum values set earlier, with Np[0] number of points
    logger.debug(f"vw = {vw}, alpha = {alpha}, Np = {Np}")

    params = (vw, alpha, const.NUC_TYPE, const.NUC_ARGS)
    sd_v = ssm.spec_den_v_bag(z, params, Np[1:], method=method)
    pow_v = ssm.pow_spec(z, sd_v)
    V2_pow_v = float(np.trapezoid(pow_v/z, z))

    sd_gw, y = ssm.spec_den_gw(z, sd_v)
    pow_gw = ssm.pow_spec(y, sd_gw)
    gw_power = float(np.trapezoid(pow_gw/y, y))

    # Power spectra to be plotted and saved.
    # If v_xi_file is given, the power spectra computed from it are appended to these.
    pow_v_list = [pow_v]
    pow_gw_list = [pow_gw]
    file_debug_arrs: list[th.FloatArr1D] = []

    if v_xi_file is not None:
        sd_v2 = ssm.spec_den_v_bag(z, params, Np[1:], v_xi_file, method=method)
        pow_v2 = ssm.pow_spec(z, sd_v2)
        V2_pow_v = float(np.trapezoid(pow_v2/z, z))

        sd_gw2, y = ssm.spec_den_gw(z, sd_v2)
        pow_gw2 = ssm.pow_spec(y, sd_gw2)
        gw_power = float(np.trapezoid(pow_gw2/y, y))

        pow_v_list.append(pow_v2)
        pow_gw_list.append(pow_gw2)
        file_debug_arrs = [sd_v2, pow_v2, sd_gw2, pow_gw2]

    nz_string = f'nz{Np[0] // 1000}k'
    nx_string = f'_nx{Np[1] // 1000}k'
    nT_string = f'_nT{Np[2]}-'
    file_suffix = f"vw{vw:3.2f}alpha{alpha}_" + const.NUC_STRING + nz_string + nx_string + nT_string

    # Now for some plotting if requested
    if save_ids[1] is not None or show:
        f1 = plt.figure(figsize=[8, 4])
        ax_v = plt.gca()

        f2 = plt.figure(figsize=[8, 4])
        ax_gw = plt.gca()

        for pow_v_i, pow_gw_i, linestyle in zip(pow_v_list, pow_gw_list, ("-", "--"), strict=False):
            ax_gw.loglog(y, pow_gw_i, color=col, linestyle=linestyle)
            ax_v.loglog(z, pow_v_i, color=col, linestyle=linestyle)

        inter_flag = abs(bubble.CS0 - vw) < 0.05  # Due intermediate power law
        plotting.plot_guide_power_laws_prace(f1, f2, z, pow_v, y, pow_gw, inter_flag=inter_flag)

        # Pretty graph 1
        pv_min, pv_max = plotting.get_yaxis_limits(utils.PSType.V, strength)
        ax_v.grid(True)
        ax_v.set_xlabel(r'$kR_*$')
        ax_v.set_ylabel(r'$\mathcal{P}_{\rm v}(kR_*)$')
        ax_v.set_ylim(pv_min, pv_max)
        ax_v.set_xlim(const.Z_MIN, const.Z_MAX)
        f1.tight_layout()

        # Pretty graph 2
        pgw_min, pgw_max = plotting.get_yaxis_limits(utils.PSType.GW, strength)
        ax_gw.grid(True)
        ax_gw.set_xlabel(r'$kR_*$')
        ax_gw.set_ylabel(r'$\Omega^\prime_{\rm gw}(kR_*)/(H_{\rm n}R_*)$')
        ax_gw.set_ylim(pgw_min, pgw_max)
        ax_gw.set_xlim(const.Z_MIN, const.Z_MAX)
        f2.tight_layout()

        # Save graphs if requested
        if save_ids[1] is not None:
            graph_file_suffix = file_suffix + save_ids[1] + '.pdf'
            f1.savefig(os.path.join(GDP, "pow_v_" + graph_file_suffix))
            f2.savefig(os.path.join(GDP, "pow_gw_" + graph_file_suffix))

        if show:
            plt.show()
        plt.close(f1)
        plt.close(f2)

    # Save data if requested
    if save_ids[0] is not None:
        data_file_suffix = file_suffix + save_ids[0] + '.txt'
        fmt = " ".join(["%.18e"] * (len(pow_v_list) + 1))
        np.savetxt(os.path.join(MDP, 'pow_v_' + data_file_suffix), np.stack((z, *pow_v_list), axis=-1), fmt=fmt)
        np.savetxt(os.path.join(MDP, 'pow_gw_' + data_file_suffix), np.stack((y, *pow_gw_list), axis=-1), fmt=fmt)

    # Now some diagnostic comparisons between real space <v^2> and Fourier space already calculated
    v_ip, w_ip, xi = bubble.sound_shell_bag(
        vw, alpha, df_dtau_ptr=bubble.DF_DTAU_PTR_BAG,
        ode_method=bubble.DEFAULT_FLUID_INTEGRATE_METHOD, cs2_ptr=bubble.CS2_BAG_SCALAR_PTR)
    # The legacy approximation w_bar = w_n is used here, as in the original ubarf_squared function.
    Ubarf2 = bubble.ubarf2(v_ip, w_ip, xi, vw, w_bar=w_ip[-1])

    logger.debug(
        "vw = %s, alpha = %s, nucleation = %s, "
        "<v^2>=%s, Ubarf2 (1 bubble) = %s, "
        "Ratio <v^2>/Ubarf2 = %s, "
        "gw power (scaled) = %s",
        vw, alpha, const.NUC_STRING, V2_pow_v, Ubarf2, V2_pow_v / Ubarf2, gw_power
    )

    if debug:
        # If v_xi_file is given, gw_power is the one computed from it.
        debug_arrs = [sd_v, pow_v, gw_power, *file_debug_arrs]
        debug_data = [Ubarf2] + [float(np.sum(arr)) for arr in debug_arrs]
        return V2_pow_v, gw_power, debug_data
    return V2_pow_v, gw_power


@tp.overload
def all_generate_ps_prace(
        save_ids: tuple[str, str] = ('', ''),
        show: bool = True,
        debug: tp.Literal[False] = False) -> tuple[list[float], list[float]]: ...


@tp.overload
def all_generate_ps_prace(
        save_ids: tuple[str, str] = ('', ''),
        show: bool = True,
        *,
        debug: tp.Literal[True]) -> tuple[list[float], list[float], list[list[float]]]: ...


def all_generate_ps_prace(
        save_ids: tuple[str, str] = ('', ''),
        show: bool = True,
        debug: bool = False) \
        -> tuple[list[float], list[float]] | tuple[list[float], list[float], list[list[float]]]:
    """
    Generate power spectra with Prace17 SSM parameters.
    Save data files and graphs.
    Returns U-bar-f^2 and GW power as tuple of lists.
    """
    method = ssm.Method.E_CONSERVING

    v2_list: list[float] = []
    Omgw_scaled_list: list[float] = []
    debug_data: list[list[float]] = []

    for vw_list, alpha, step_list, path, dir_list in \
            zip(VW_LIST_ALL, const.ALPHA_LIST_ALL, STEP_LIST_ALL, PATH_LIST_ALL, DIR_LIST_ALL, strict=False):
        for vw, step, dir_name in zip(vw_list, step_list, dir_list, strict=False):
            v_xi_file = PATH_HEAD + path + dir_name + FILE_PATTERN.format(step)
            logger.debug("v_xi_file: %s", v_xi_file)
            if debug:
                v2, Omgw, data = generate_ps(vw, alpha, method, v_xi_file, save_ids, show, debug=True)
                debug_data.append(data)
            else:
                v2, Omgw = generate_ps(vw, alpha, method, v_xi_file, save_ids, show)
            v2_list.append(v2)
            Omgw_scaled_list.append(Omgw)

    if debug:
        return v2_list, Omgw_scaled_list, debug_data
    return v2_list, Omgw_scaled_list
