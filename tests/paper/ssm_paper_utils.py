"""Compare SSM prediction with data
Creates and plots velocity and GW power spectra from SSM

Modified from
https://bitbucket.org/hindmars/sound-shell-model/
"""

# import concurrent.futures as fut
import io
import logging
# import os
import typing as tp

import numba
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from pttools import bubble
from pttools import ssmtools as ssm
from tests.paper import const
from tests.paper import plotting
from tests.paper import utils
import tests.paper.tex_utils as tu
from tests.utils.const import TEST_DATA_PATH

logger = logging.getLogger(__name__)

# bubble.setup_plotting()

# Model data path
MD_PATH = TEST_DATA_PATH + "/"

# All run parameters

VW_INTER_LIST = [0.92, 0.80, 0.731, 0.56, 0.44]

# NB1 prace runs did not include intermediate 0.80, 0.44.
# NB2 prace runs had high-T approximation to effective potential, not bag.
# NB3 prace runs had nominal vw = 0.72, but this is hybrid in bag EOS.

# t_weak_list = [1210, 1380, 1630, 1860, 2520]
# t_inter_list = [1180, 1480, 2650]

# dt_weak_list = [0.08*t for t in t_weak_list]
# dt_inter_list = [0.08*t for t in t_inter_list]
# dt_inter_list[1]=0.05*t_inter_list[1] # dx=1

# step_weak_list = [int(round(dt/20)*20) for dt in dt_weak_list]
# step_inter_list = [int(round(dt/20)*20) for dt in dt_inter_list]

# dir_weak_list = [
#     'results-weak-scaled_etatilde0.19_v0.92_dx2/',
#     'results-weak-scaled_etatilde0.35_v0.80_dx2/',
#     'results-weak-scaled_etatilde0.51_v0.68_dx2/',
#     'results-weak-scaled_etatilde0.59_v0.56_dx2/',
#     'results-weak-scaled_etatilde0.93_v0.44_dx2/'
# ]
# dir_inter_list = [
#     'results-intermediate-scaled_etatilde0.17_v0.92_dx2/',
#     'results-intermediate-scaled_etatilde0.40_v0.72_dx1/',
#     'results-intermediate-scaled_etatilde0.62_v0.44_dx2/'
# ]

# path_head = 'fluidprofiles/'
# path_list_all = ['weak/', 'intermediate/']
# file_pattern = 'data-extracted-with-e.{:05d}.txt'

VW_LIST_ALL = [const.VW_WEAK_LIST, VW_INTER_LIST]
# step_list_all = [step_weak_list, step_inter_list]
# dir_list_all = [dir_weak_list, dir_inter_list]

# Files used in prace paper 2017
# file_list_weak = [
#     'eta-0.19-t-1212.5-Nb84.txt',
#     'eta-0.35-t-1375.0-Nb84.txt',
#     'eta-0.51-t-1625.0-Nb84.txt',
#     'eta-0.59-dx1-t-1862.5-Nb11.txt',
#     # 'eta-0.59-t-1875.0-Nb84.txt',
#     # 'eta-0.59-t-475.0-Nb5376.txt',
#     'eta-0.93-t-2525.0-Nb84.txt'
# ]
# file_list_inter = [
#     'eta-inter-0.17-t-1175.0-Nb84.txt',
#     'eta-inter-0.40-dx1-t-1475.0-Nb11.txt',
#     'eta-inter-0.62-t-2650.0-Nb84.txt'
# ]


# Functions

@numba.njit
def cwg_fitfun(k, p0, p1):
    return p0*np.power(k/p1, 3.0)*np.power(7.0/(4.0 + 3.0*np.power(k/p1, 2.0)), 7.0/2.0)


@numba.njit
def double_broken_power_law(z, A, z0, z1, a, b, c, d=4., e=2.):
    s = z/z1
    D = z1/z0
    Dpow = D**d
    norm = (1 + Dpow)**((a-b)/d) * (b-c)**((b-c)/e)
    m = a - (a-b)*Dpow/(1 + Dpow)
    # with this norm, A is the peak power
    return A * norm*s**a/(1 + (D*s)**d)**((a-b)/d) / (b-c-m + m*s**e)**((b-c)/e)


@numba.njit
def ssm_fitfun(z, A, z0, z1):
    return double_broken_power_law(z, A, z0, z1, 9, 1, -4)


def get_cwg_fit_pars(y, pow_gw):
    frange = np.where(y > 10)
    pars0 = (pow_gw[frange][0], 10)
    pars, _ = curve_fit(cwg_fitfun, y[frange], pow_gw[frange], pars0)
    return pars[0], pars[1]


def get_ssm_fit_pars(y, pow_gw):
    frange = np.where(y > 1e-8)
    pars0 = (pow_gw[frange][0], 3, 20)
    pars, _ = curve_fit(ssm_fitfun, y[frange], pow_gw[frange], pars0, sigma=np.sqrt(pow_gw[frange]))
    return pars


def add_cwg_fit(f_gw, y, pow_gw):
    p = get_cwg_fit_pars(y, pow_gw)
    pow_gw_sim_cwg = cwg_fitfun(y, p[0], p[1])
    f_gw.axes[0].loglog(y, pow_gw_sim_cwg, 'k-.', label='CWG fit')
    f_gw.axes[0].legend(loc='best')
    return p


def add_ssm_fit(f_gw, y, pow_gw):
    p = get_ssm_fit_pars(y, pow_gw)
    pow_gw_sim_ssm = ssm_fitfun(y, p[0], p[1], p[2])
    f_gw.axes[0].loglog(y, pow_gw_sim_ssm, 'k--', label='SSM fit')
    f_gw.axes[0].legend(loc='best')
    return p


def make_1dh_compare_table(params_list, v2_list,
                           file_name: tp.Union[str, io.TextIOBase] = 'table_1dh_compare.tex'):

    if isinstance(file_name, io.TextIOBase):
        f = file_name
    else:
        f = open(file_name, 'w')
    f.write('\\begin{tabular}{cc | rrr }\n')

    f.write('\\hline\\hline\n')
    f.write('$10^2\\alpha$' + ' & ')
    f.write('$v_{\\rm w}$' + ' & ')
    f.write('$\\bar{U}_{f,3}^{\\rm sim}$' + ' & ')
    f.write('$\\bar{U}_{f,3}^{\\rm exp}$' + ' & ')
    f.write('$\\bar{U}_{f,3}^{\\rm 1d}$' + ' \\\\ \n')
    # f.write()
    f.write('\\hline\n')

    for n, params in enumerate(params_list):
        alpha = params[0]
        vw = params[1]
        v2_sim = v2_list[n][0]
        v2_exp = v2_list[n][1]
        Ubarf_1d_ssm = np.sqrt(bubble.get_ubarf2(vw, alpha))

        f.write(tu.tex_sf(100*alpha) + ' & ')
        f.write(f'{vw:.2f} & ')
        f.write(f'{np.sqrt(v2_sim)*1000:4.1f} & ')
        f.write(f'{np.sqrt(v2_exp)*1000:4.1f} & ')
        f.write(f'{1000*Ubarf_1d_ssm:4.1f}'.replace('nan', '   ') + ' \\\\ \n')
        # f.write()

    f.write('\\hline\\hline\n')
    f.write('\\end{tabular}\n')
    if not isinstance(file_name, io.TextIOBase):
        f.close()

    return None


def make_3dh_compare_table(params_list, v2_list, Omgw_list, p_list,
                           file_name: tp.Union[str, io.TextIOBase] = 'table_3dh_compare.tex') -> None:
    """
    Prints table to file, comparing selected statistics between
    SSM and "Prace" 3dh hydro simulations (Hindmarsh et al 2017)
    Mean square fluid velocity.

    The results are precomputed, and therefore this function doesn't call pttools.
    """
    Ubarf_prace = [
        4.60e-3, 5.75e-3, 8.65e-3, 13.8e-3, 7.51e-3,
        43.7e-3, np.nan, 65.0e-3, np.nan, 54.5e-3]
    # Acoustic Gravitational wave production efficiency
    Om_tilde_prace = [
        1.2e-2, 1.4e-2, 0.62e-2, 0.32e-2, 1.1e-2,
        2.0e-2, np.nan, 1.8e-2, np.nan, 1.7e-2]
    # Amplitude of fit to cwg CWG form for GW PS
    A_prace = [
        1.4e-11, 3.1e-11, 8.1e-11, np.nan, 8.2e-11,
        1.6e-7, np.nan, 3.7e-7, np.nan, 4.3e-7]
    # Peak z = kR* of GW PS
    z_peak_prace = [
        8.6, 10.4, 18.3, np.nan, 9.9,
        8.5, np.nan, 16.1, np.nan, 6.9]

    if isinstance(file_name, io.TextIOBase):
        f = file_name
    else:
        f = open(file_name, 'w')
    f.write('\\begin{tabular}{cc | rr | rr | ll | rr}\n')

    f.write('\\hline\\hline\n')
    f.write('$10^2\\alpha$' + ' & ')
    f.write('$v_{\\rm w}$' + ' & ')
    f.write('$\\bar{U}_{f,3}^{\\rm sim}$' + ' & ')
    f.write('$\\bar{U}_{f,3}^{\\rm 3dh}$' + ' & ')
    # f.write('$\\bar{U}_{f,3}^{\\rm 1d}$' + ' & ')
    f.write('$\\tilde\\Omega_{\\rm gw,2}^{\\rm sim}$' + ' & ')
    f.write('$\\tilde\\Omega_{\\rm gw,2}^{\\rm 3dh}$' + ' & ')
    f.write('\\hspace{1em} $A^{\\rm sim}$' + ' & ')
    f.write('\\hspace{1em} $A^{\\rm 3dh}$' + ' & ')
    f.write('$x_{\\rm p}^{\\rm sim}$' + ' & ')
    f.write('$x_{\\rm p}^{\\rm 3dh}$' + ' \\\\ \n')
    # f.write()
    f.write('\\hline \n')

    # for n, params, v2, Omgw, p in enumerate(zip(params_list, v2_list, Omgw_list, p_list)):
    for n, params in enumerate(params_list):
        alpha = params[0]
        vw = params[1]
        v2_sim = v2_list[n][0]
        # Ubarf_1d_ssm = np.sqrt(b.get_ubarf2(vw,alpha))

        Omgw_sim = Omgw_list[n][0]
        A = p_list[n][0]
        z_peak = p_list[n][1]

        ad_ind = 4/(3*(1+alpha))
        Om_tilde = Omgw_sim/(3*(ad_ind*v2_sim)**2)

        # if alpha < 0.01:
        #     factor = 1e11
        # elif alpha < 0.1:
        #     factor = 1e7

        # f.write('{alpha:6} & ')
        f.write(tu.tex_sf(100*alpha) + ' & ')
        f.write(f'{vw:.2f} & ')
        f.write(f'{1000*np.sqrt(v2_sim):4.1f} & ')
        f.write(f'{1000*Ubarf_prace[n]:4.1f}'.replace('nan', '   ') + ' & ')
        # f.write(f'{1000*Ubarf_1d_ssm:4.1f}'.replace('nan','   ') + ' & ')
        f.write(f'{100*Om_tilde:3.1f} & ')
        f.write(f'{100*Om_tilde_prace[n]:3.1f}'.replace('nan', '   ') + ' & ')
        # f.write(f'{A:.1f} & ')
        # f.write(f'{A_prace[n]:.1f}'.replace('nan',8*' ') + ' & ')
        f.write(tu.tex_sf(A, mult='\\cdot') + ' & ')
        if A_prace[n] is not np.nan:
            f.write(tu.tex_sf(A_prace[n], mult='\\cdot') + ' & ')
        else:
            f.write(8*' ' + ' & ')
        f.write(f'{z_peak:4.1f} & ')
        f.write(f'{z_peak_prace[n]:4.1f}'.replace('nan', '   ') + ' \\\\ \n')
        # f.write()

    f.write('\\hline\\hline \n')
    f.write('\\end{tabular} \n')

    if not isinstance(file_name, io.TextIOBase):
        f.close()


def make_nuc_compare_table(params_list, v2_list, Omgw_list, p_sim_list, p_exp_list,
                           file_name: tp.Union[str, io.TextIOBase] = 'table_nuc_compare.tex') -> None:
    """
    Prints table to stdout, displaying selected statistics
    comparing between simulataneous and exponential nucleation.

    The results are precomputed, and therefore this function doesn't call pttools.
    """
    # print('\\begin{tabular}{cc | rr | rr | ll | rr | rr}')
    if isinstance(file_name, io.TextIOBase):
        f = file_name
    else:
        f = open(file_name, 'w')
    f.write('\\begin{tabular}{cc | rr | ll | rr | rr}\n')

    f.write('\\hline\\hline\n')
    f.write('$10^2\\alpha$' + ' & ')
    f.write('$v_{\\rm w}$' + ' & ')
    # f.write('$\\bar{U}_{f,3}^{\\rm sim}$' + ' & ')
    # f.write('$\\bar{U}_{f,3}^{\\rm exp}$' + ' & ')
    f.write('$\\tilde\\Omega_{\\rm gw,2}^{\\rm sim}$' + ' & ')
    f.write('$\\tilde\\Omega_{\\rm gw,2}^{\\rm exp}$' + ' & ')
    f.write('\\hspace{1em} $A^{\\rm sim}$' + ' & ')
    f.write('\\hspace{1em} $A^{\\rm exp}$' + ' & ')
    f.write('$x_{\\rm p}^{\\rm sim}$' + ' & ')
    f.write('$x_{\\rm p}^{\\rm exp}$' + ' & ')
    f.write('$x_{\\rm b}^{\\rm exp}$' + ' & ')
    f.write('$x_{\\rm p}^{\\rm exp}\\De_\\text{w}$' + ' \\\\ \n')
    # f.write()
    f.write('\\hline\n')

    # for n, params, v2, Omgw, p in enumerate(zip(params_list, v2_list, Omgw_list, p_list)):
    for n, params in enumerate(params_list):
        alpha = params[0]
        vw = params[1]
        v2_sim = v2_list[n][0]
        v2_exp = v2_list[n][1]
        Omgw_sim = Omgw_list[n][0]
        Omgw_exp = Omgw_list[n][1]
        A_sim = p_sim_list[n][0]
        A_exp = p_exp_list[n][0]
        z_peak_sim = p_sim_list[n][1]
        z_peak_exp = p_exp_list[n][2]
        z_break_exp = abs(p_exp_list[n][1])

        ad_ind = 4/(3*(1+alpha))
        Om_tilde_sim = Omgw_sim/(3*(ad_ind*v2_sim)**2)
        Om_tilde_exp = Omgw_exp/(3*(ad_ind*v2_exp)**2)

        # if alpha < 0.01:
        #     factor = 1e11
        # elif alpha < 0.1:
        #     factor = 1e7

        # f.write(f'{100*alpha:6.2f} & ')
        f.write(tu.tex_sf(100*alpha) + ' & ')
        f.write(f'{vw:.2f} & ')
        # f.write(f'{np.sqrt(v2_sim)*1000:4.1f} & ')
        # f.write(f'{np.sqrt(v2_exp)*1000:4.1f} & ')
        f.write(f'{100*Om_tilde_sim:3.1f} & ')
        f.write(f'{100*Om_tilde_exp:3.1f} & ')
        # f.write(f'{A:.1f} & ')
        # f.write(f'{A_prace[n]:.1f}'.replace('nan',8*' ') + ' & ')
        f.write(tu.tex_sf(A_sim, mult='\\cdot') + ' & ')
        f.write(tu.tex_sf(A_exp, mult='\\cdot') + ' & ')
        f.write(f'{z_peak_sim:4.1f} & ')
        f.write(f'{z_peak_exp:4.1f} & ')
        f.write(f'{z_break_exp:3.1f} & ')
        f.write(f'{z_peak_exp * abs(vw - bubble.CS0) / vw :3.1f} \\\\ \n')
        # f.write()

    f.write('\\hline\\hline\n')
    f.write('\\end{tabular} \n')
    if not isinstance(file_name, io.TextIOBase):
        f.close()


def save_compare_nuc_data(file, params_list, v2_list, Omgw_list, p_cwg_list, p_ssm_list):
    data = []
    for params, v2, Omgw, pc, ps in zip(params_list, v2_list, Omgw_list, p_cwg_list, p_ssm_list):
        data.append(params + v2 + Omgw + pc + ps)

    np.savetxt(file, data)

    return data


def load_compare_nuc_data(file: str):
    data = np.loadtxt(file)
    params_list = []
    v2_list = []
    Omgw_list = []
    p_cwg_list = []
    p_ssm_list = []

    params = data[:, 0:2]
    v2 = data[:, 2:4]
    Omgw = data[:, 4:6]
    p_cwg = data[:, 6:8]
    p_ssm = data[:, 8:12]

    for n, p in enumerate(params):
        params_list.append(list(p))
        v2_list.append(list(v2[n]))
        Omgw_list.append(list(Omgw[n]))
        p_cwg_list.append(list(p_cwg[n]))
        p_ssm_list.append(list(p_ssm[n]))

    return params_list, v2_list, Omgw_list, p_cwg_list, p_ssm_list


def ps_from_ssm(
        vw,
        alpha,
        nuc_type: ssm.NucType = ssm.NucType.SIMULTANEOUS,
        nuc_args=(1.,),
        Np=const.NP_LIST[-1],
        method: ssm.Method = ssm.Method.E_CONSERVING):
    """Get velocity and GW power spectra from SSM"""

    nuc_string = nuc_type[0:3] + '_'
    for n in range(len(nuc_args)):
        nuc_string += str(nuc_args[n]) + '_'

    z = np.logspace(np.log10(const.Z_MIN), np.log10(const.Z_MAX), Np[0])

    sd_v = ssm.spec_den_v(z, [vw, alpha, nuc_type, nuc_args], Np[1:], method=method)
    pow_v = ssm.pow_spec(z, sd_v)

    V2_pow_v = np.trapz(pow_v/z, z)

    sd_gw, y = ssm.spec_den_gw_scaled(z, sd_v)
    pow_gw = ssm.pow_spec(y, sd_gw)

    gw_power = np.trapz(pow_gw/y, y)

    Ubarf = np.sqrt(V2_pow_v)
    AdInd = 4/(3*(1+alpha))
    Omgwtil = gw_power/(AdInd*V2_pow_v)**2

    logger.debug(f"{nuc_string:s} {alpha} {vw:.2f} {V2_pow_v:.3e} {1000*Ubarf:5.2f} {gw_power:.3e} {100*Omgwtil:.3f}")

    return z, pow_v, y, pow_gw


def plot_ps_compare_res(
        vw,
        alpha: float,
        nuc_type: ssm.NucType = ssm.NucType.SIMULTANEOUS,
        nuc_args=(1.,),
        save_id: str = None,
        graph_file_type=None,
        method: ssm.Method = ssm.Method.E_CONSERVING):
    """
    Plots power spectra predictions of SSM with different resolutions in Np_list
    Saves data and graphs if save_id is set
    """
    strength = utils.Strength.WEAK
    if alpha >= 0.05:
        strength = utils.Strength.INTER
    if alpha >= 0.5:
        strength = utils.Strength.STRONG

    z_list = []
    pow_v_list = []
    y_list = []
    pow_gw_list = []

    nuc_string = nuc_type[0:3] + '_'
    for n in range(len(nuc_args)):
        nuc_string += str(nuc_args[n]) + '_'

    nz_string_all = 'nz'
    nx_string_all = '_nx'
    nT_string_all = '_nT'

    for Np in const.NP_LIST:
        z, pow_v, y, pow_gw = ps_from_ssm(vw, alpha, nuc_type, nuc_args, Np, method)
        z_list.append(z)
        pow_v_list.append(pow_v)
        y_list.append(y)
        pow_gw_list.append(pow_gw)

        if save_id is not None:

            nz_string = f'nz{Np[0] // 1000}k'
            nx_string = f'_nx{Np[1] // 1000}k'
            nT_string = f'_nT{Np[2]}-'

            data_file_suffix = f"vw{vw}alpha{alpha}_" + nuc_string \
                + nz_string + nx_string + nT_string + save_id + '.txt'
            np.savetxt(MD_PATH + 'pow_v_' + data_file_suffix, np.stack((z, pow_v), axis=-1), fmt='%.18e %.18e')
            np.savetxt(MD_PATH + 'pow_gw_' + data_file_suffix, np.stack((y, pow_gw), axis=-1), fmt='%.18e %.18e')

        else:
            save_id = ''

        nz_string_all += f'{Np[0] // 1000}k'
        nx_string_all += f'{Np[1] // 1000}k'
        nT_string_all += f'{Np[2]}-'

    with plt.rc_context({'legend.fontsize': 12}):
        f_v = plotting.plot_ps(
            z_list, pow_v_list, utils.PSType.V, ax_limits=strength, col_list=const.COLOURS)
        f_gw = plotting.plot_ps(
            y_list, pow_gw_list, utils.PSType.GW, ax_limits=strength, col_list=const.COLOURS)

        # Now plot guide power laws
        if method is ssm.Method.E_CONSERVING:
            nv_lo = 5
            ngw_lo = 9
        else:
            nv_lo = 3
            ngw_lo = 5

        inter_flag = (abs(bubble.CS0 - vw) < 0.05)
        plotting.plot_guide_power_laws_prace(
            f_v, f_gw, z_list[0], pow_v_list[0], y_list[0], pow_gw_list[0],
            (nv_lo, ngw_lo), inter_flag)

    # Save graph if asked for
    if save_id is not None:
        graph_file_suffix = f"vw{vw:.2f}alpha{alpha}_" + nuc_string \
                            + nz_string_all + nx_string_all + nT_string_all + save_id + '.' + const.FILE_TYPE
        f_v.savefig("pow_v_" + graph_file_suffix)
        f_gw.savefig("pow_gw_" + graph_file_suffix)

    plt.show()

    return f_v, f_gw


def plot_ps_1bubble(
        vw,
        alpha: float,
        save_id: str = None,
        graph_file_type: str = None,
        Np=const.NP_LIST[-1],
        debug: bool = False) -> tp.Union[plt.Figure, tp.Tuple[plt.Figure, np.ndarray]]:
    """
    Plots power spectra predictions of 1 bubble. Shown are
    |A|^2, |f'(z)|^2/2 and |l(z)|^2/2
    Saves data if save_id is set
    Saves graph file if graph_file_type is set
    """
    strength = utils.Strength.WEAK
    if alpha >= 0.05:
        strength = utils.Strength.INTER
    if alpha >= 0.5:
        strength = utils.Strength.STRONG

    z = np.logspace(np.log10(const.Z_MIN), np.log10(const.Z_MAX), Np[0])

    nz_string = f'nz{Np[0] // 1000}k_'
    nx_string = f'nx{Np[1] // 1000}k-'

    A2, fp2_2, lam2 = ssm.A2_e_conserving(z, vw, alpha, npt=Np[1:])

    z_list = 3*[z]
    ph_sp_fac = z**3/(2*np.pi**2)
    ps_list = [ph_sp_fac * A2, ph_sp_fac * fp2_2 / 2, ph_sp_fac * bubble.CS0_2 * lam2 / 4]
    leg_list = ['$|A|^2$', r'$|f^\prime(z)|^2/4$', '$c_{\\rm s}^2|l(z)|^2/4$']

    f = plotting.plot_ps(
        z_list, ps_list, utils.PSType.UNKNOWN, ax_limits=strength,
        col_list=const.COLOURS, leg_list=leg_list)

    inter_flag = (abs(bubble.CS0 - vw) < 0.05)
    plotting.plot_guide_power_laws_ssm(f, z, ph_sp_fac*A2, utils.PSType.V, inter_flag=inter_flag)

    if save_id is None:
        save_id = ''

    # Save graph if asked for
    if graph_file_type is not None:
        graph_file_suffix = f"vw{vw:.2f}alpha{alpha}_"  \
            + nz_string + nx_string + save_id + '.' + graph_file_type
        f.savefig(MD_PATH + "one_bub_" + graph_file_suffix)

    # plt.show()

    if debug:
        return f, np.array([z, A2, fp2_2, lam2])
    return f


def plot_ps_compare_nuc(
        vw: float,
        alpha: float,
        save_id: str = None,
        graph_file_type: str = None) -> tp.Tuple[list, list, list, list]:
    """
    Plots power spectra predictions of SSM with different nucleation models
    Saves data if save_id is set.
    Saves graph file if graph_file_type is set.
    """
    method = ssm.Method.E_CONSERVING
    nuc_type_list = [ssm.NucType.SIMULTANEOUS, ssm.NucType.EXPONENTIAL]
    nuc_args_list = [(1.,), (1.,)]

    strength = utils.Strength.WEAK
    if alpha >= 0.05:
        strength = utils.Strength.INTER
    if alpha >= 0.5:
        strength = utils.Strength.STRONG

    z_list = []
    pow_v_list = []
    y_list = []
    pow_gw_list = []

    v2_list = []
    Omgw_scaled_list = []

    Np = const.NP_LIST[-1]

    nz_string = f'nz{Np[0] // 1000}k'
    nx_string = f'_nx{Np[1] // 1000}k'
    nT_string = f'_nT{Np[2]}-'

    nuc_string_all = ''

    for nuc_type, nuc_args in zip(nuc_type_list, nuc_args_list):
        z, pow_v, y, pow_gw = ps_from_ssm(vw, alpha, nuc_type, nuc_args, Np, method)
        z_list.append(z)
        pow_v_list.append(pow_v)
        y_list.append(y)
        pow_gw_list.append(pow_gw)

        nuc_string = nuc_type[0:3] + '_'
        for n in range(len(nuc_args)):
            nuc_string += str(nuc_args[n]) + '_'

        if save_id is not None:
            data_file_suffix = f"vw{vw:.2f}alpha{alpha}_" + nuc_string \
                + nz_string + nx_string + nT_string + save_id + '.txt'
            np.savetxt(MD_PATH + 'pow_v_' + data_file_suffix, np.stack((z, pow_v), axis=-1), fmt='%.18e %.18e')
            np.savetxt(MD_PATH + 'pow_gw_' + data_file_suffix, np.stack((y, pow_gw), axis=-1), fmt='%.18e %.18e')
        else:
            save_id = ''

        nuc_string_all += nuc_string
        v2_list.append(np.trapz(pow_v/z, z))
        Omgw_scaled_list.append(np.trapz(pow_gw/y, y))

    f_v = plotting.plot_ps(
        z_list, pow_v_list, utils.PSType.V,
        ax_limits=strength, col_list=const.COLOURS, leg_list=nuc_type_list)
    f_gw = plotting.plot_ps(
        y_list, pow_gw_list, utils.PSType.GW,
        ax_limits=strength, col_list=const.COLOURS, leg_list=nuc_type_list)

    inter_flag = (abs(bubble.CS0 - vw) < 0.05)
    plotting.plot_guide_power_laws_prace(
        f_v, f_gw, z_list[0], pow_v_list[0],
        y_list[0], pow_gw_list[0], inter_flag=inter_flag)

    p_cwg = add_cwg_fit(f_gw, y_list[0], pow_gw_list[0])
    p_ssm = add_ssm_fit(f_gw, y_list[1], pow_gw_list[1])

    if save_id is None:
        save_id = ''

    # Save graph if asked for
    if graph_file_type is not None:
        graph_file_suffix = f"vw{vw:.2f}alpha{alpha}_" + nuc_string_all \
            + nz_string + nx_string + nT_string + save_id + '.' + graph_file_type
        f_v.savefig(MD_PATH + "pow_v_" + graph_file_suffix)
        f_gw.savefig(MD_PATH + "pow_gw_" + graph_file_suffix)

    # plt.show()

    return v2_list, Omgw_scaled_list, list(p_cwg), list(p_ssm)


# def plot_ps_from_1dhydro(nt_list, v_xi_file_path, method='e_conserving', save_id=None, graph_file_suffix=None,
#                         file_pattern='data-extracted-with-e.{:05}.txt'):
#    # Calls ps_from_1dhydro to calculate velocity and GW power spectra in SSM
#    # then calls plot_ps to plot them
#    # Figures out whether it's a weak or intermediate transition from filename
#    # Optional save of data and graph if save_id (extra string for data file name e.g. _foo)
#    # or graph_file_suffix set (e.g. '_foo.pdf')
#    # Returns figure handles
#    leg_list = []
#    ls_list = []
#
#    pt_type=None
#    if 'weak' in v_xi_file_path:
#        pt_type = 'weak'
#    elif 'inter' in v_xi_file_path:
#        pt_type = 'inter'
#
#    z_list, pow_v_list, y_list, pgw_list, t_list = ps_from_1dhydro(nt_list, v_xi_file_path, file_pattern)
#
#    for t in t_list:
#        leg_list.append('{:6.1f}'.format(t))
#        ls_list.append('--')
#
#    fig_v = plot_ps(z_list, pow_v_list, 'v', ax_limits=pt_type,
#                    leg_list=leg_list, ls_list=ls_list)
#    fig_gw = plot_ps(y_list, pgw_list, 'gw', ax_limits=pt_type,
#                     leg_list=leg_list, ls_list=ls_list)
#
#    if save_id is not None:
#        data_file = make_file_string_1dh(v_xi_file_path)
#
#        for z, pv, nt, t in zip(z_list, pow_v_list, nt_list, t_list):
#            x1 = np.concatenate(([nt,],z))
#            x2 = np.concatenate(([t,],pv))
#            np.savetxt(md_path + 'pow_v_1dh_' + data_file + '_nt{:05}'.format(nt)
#                + save_id + '.txt', np.stack((x1,x2),axis=-1), fmt='%.18e %.18e')
#
#        for y, pgw, nt, t in zip(y_list, pgw_list, nt_list, t_list):
#            x1 = np.concatenate(([nt,],y))
#            x2 = np.concatenate(([t,],pgw))
#            np.savetxt(md_path + 'pow_gw_1dh_' + data_file + '_nt{:05}'.format(nt)
#                + save_id + '.txt', np.stack((x1,x2), axis=-1), fmt='%.18e %.18e')
#
#
#    if graph_file_suffix is not None:
#        graph_file = make_file_string_1dh(v_xi_file_path)
#        fig_v.savefig('pow_v_1dh_' + graph_file + graph_file_suffix)
#        fig_gw.savefig('pow_gw_1dh_' + graph_file + graph_file_suffix)
#
#    return fig_v, fig_gw


# def do_all_1dhydro(nt_list=[100,150], save_id='', graph_file_type='.pdf'):
#    # Calculates, plots and saves v and GW power spectra from 1d hydro
#    # bubble profiles in SSM, at timesteps specified by nt_list.
#    # Probably should allow a list of lists of nt.
#    weak_list = ['results-weak-scaled_etatilde0.19_v0.92_dx2/',
#                 'results-weak-scaled_etatilde0.35_v0.80_dx2/',
#                 'results-weak-scaled_etatilde0.51_v0.68_dx2/',
#                 'results-weak-scaled_etatilde0.59_v0.56_dx2/',
#                 'results-weak-scaled_etatilde0.93_v0.44_dx2/']
#    inter_list = ['results-intermediate-scaled_etatilde0.17_v0.92_dx2/',
#                 'results-intermediate-scaled_etatilde0.40_v0.72_dx1/',
#                 'results-intermediate-scaled_etatilde0.62_v0.44_dx2/']
#
#    weak_path = 'fluidprofiles/weak/'
#    inter_path = 'fluidprofiles/intermediate/'
#
#    f_v_list = []
#    f_gw_list = []
#
#    for flist, path in zip([weak_list, inter_list], [weak_path, inter_path]):
#        for p in flist:
#            v_xi_file_path = path + p
#            f_v, f_gw = plot_ps_from_1dhydro(nt_list, v_xi_file_path, save_id=save_id,
#                                             graph_file_suffix=graph_file_type)
#            f_v_list.append(f_v)
#            f_gw_list.append(f_gw)
#
#    return f_v_list, f_gw_list


def plot_and_save(vw, alpha, method='e_conserving', v_xi_file=None, suffix=None):
    """
    Plots the Velocity power spectrum as a function of $kR_*$.
    Plots the scaled GW power spectrum as a function of $kR_*$.
    Saves power spectra in files pow_v_*, pow_gw_* if suffix is set.
    """

    Np = const.NP_LIST[-1]
    col = const.COLOURS[0]

    if alpha < 0.01:
        strength = utils.Strength.WEAK
    elif alpha < 0.1:  # intermediate transition
        strength = utils.Strength.INTER
    else:
        logger.warning("alpha > 0.1, taking strength = inter")

    f1 = plt.figure(figsize=[8, 4])
    ax_v = plt.gca()

    f2 = plt.figure(figsize=[8, 4])
    ax_gw = plt.gca()

    V2_pow_v = []
    gw_power = []

    z = np.logspace(np.log10(const.Z_MIN), np.log10(const.Z_MAX), Np[0])

    # Array using the minimum & maximum values set earlier, with Np[0] number of points
    logger.debug(f"vw = {vw}, alpha = {alpha}, Np = {Np}")

    sd_v = ssm.spec_den_v(z, [vw, alpha, const.NUC_TYPE, const.NUC_ARGS], Np[1:], method=method)
    pow_v = ssm.pow_spec(z, sd_v)
    ax_v.loglog(z, pow_v, color=col)
    V2_pow_v.append(np.trapz(pow_v/z, z))

    if v_xi_file is not None:
        sd_v2 = ssm.spec_den_v(z, [vw, alpha, const.NUC_TYPE, const.NUC_ARGS], Np[1:], v_xi_file, method=method)
        pow_v2 = ssm.pow_spec(z, sd_v2)
        ax_v.loglog(z, pow_v2, color=col, linestyle='--')
        V2_pow_v.append(np.trapz(pow_v2/z, z))

    sd_gw, y = ssm.spec_den_gw_scaled(z, sd_v)
    pow_gw = ssm.pow_spec(y, sd_gw)

    ax_gw.loglog(y, pow_gw, color=col)
    gw_power.append(np.trapz(pow_gw/y, y))

    if v_xi_file is not None:
        sd_gw2, y = ssm.spec_den_gw_scaled(z, sd_v2)
        pow_gw2 = ssm.pow_spec(y, sd_gw2)
        ax_gw.loglog(y, pow_gw2, color=col, linestyle='--')
        gw_power.append(np.trapz(pow_gw2/y, y))

    inter_flag = (abs(bubble.CS0 - vw) < 0.05)  # Due intermediate power law
    plotting.plot_guide_power_laws_prace(f1, f2, z, pow_v, y, pow_gw, inter_flag=inter_flag)

    # Pretty graph 1
    pv_min, pv_max = plotting.get_yaxis_limits(utils.PSType.V, strength)
    ax_v.grid(True)
    ax_v.set_xlabel(r'$kR_*$')
    ax_v.set_ylabel(r'$\mathcal{P}_{\rm v}(kR_*)$')
    ax_v.set_ylim([pv_min, pv_max])
    ax_v.set_xlim([const.Z_MIN, const.Z_MAX])
    f1.tight_layout()

    # Pretty graph 2
    pgw_min, pgw_max = plotting.get_yaxis_limits(utils.PSType.GW, strength)
    ax_gw.grid(True)
    ax_gw.set_xlabel(r'$kR_*$')
    ax_gw.set_ylabel(r'$\Omega_{\rm gw}(kR_*)$')
    ax_gw.set_ylim([pgw_min, pgw_max])
    ax_gw.set_xlim([const.Z_MIN, const.Z_MAX])
    f2.tight_layout()

    if suffix is not None:
        nz_string = f'nz{Np[0] // 1000}k'
        nx_string = f'_nx{Np[1] // 1000}k'
        nT_string = f'_nT{Np[2]}-'
        file_suffix = f"vw{vw:3.2f}alpha{alpha}_" + const.NUC_STRING \
                      + nz_string + nx_string + nT_string + suffix

        data_file_suffix = file_suffix + '.txt'
        graph_file_suffix = file_suffix + '.pdf'

        if v_xi_file is None:
            np.savetxt(MD_PATH + 'pow_v_' + data_file_suffix,
                       np.stack((z, pow_v), axis=-1), fmt='%.18e %.18e')
            np.savetxt(MD_PATH + 'pow_gw_' + data_file_suffix,
                       np.stack((y, pow_gw), axis=-1), fmt='%.18e %.18e')
        else:
            np.savetxt(MD_PATH + 'pow_v_' + data_file_suffix,
                       np.stack((z, pow_v, pow_v2), axis=-1), fmt='%.18e %.18e %.18e')
            np.savetxt(MD_PATH + 'pow_gw_' + data_file_suffix,
                       np.stack((y, pow_gw, pow_gw2), axis=-1), fmt='%.18e %.18e %.18e')
        f1.savefig(MD_PATH + "pow_v_" + graph_file_suffix)
        f2.savefig(MD_PATH + "pow_gw_" + graph_file_suffix)

    # Now some comparisons between real space <v^2> and Fourier space already calculated
    v_ip, w_ip, xi = bubble.fluid_shell(vw, alpha)
    Ubarf2 = bubble.Ubarf_squared(v_ip, w_ip, xi, vw)

    logger.debug(
        f"vw = {vw}, alpha = {alpha}, nucleation = {const.NUC_STRING}, "
        f"<v^2> = {V2_pow_v}, Ubarf2 (1 bubble) = {Ubarf2}, "
        f"Ratio <v^2>/Ubarf2 = {[V2/Ubarf2 for V2 in V2_pow_v]}, "
        f"gw power (scaled) = {gw_power}"
    )

    plt.show()
    return V2_pow_v, gw_power


def do_all_plot_ps_compare_nuc(save_id: str = None, graph_file_type=None):
    v2_list = []
    Omgw_scaled_list = []

    param_list = []
    p_cwg_list = []
    p_ssm_list = []

    # This loop cannot be multi-threaded, as Matplotlib is not thread-safe
    for vw_list, alpha, in zip(VW_LIST_ALL, const.ALPHA_LIST_ALL):
        for vw in vw_list:
            v2, Omgw_scaled, p_cwg, p_ssm = plot_ps_compare_nuc(vw, alpha, save_id, graph_file_type)
            v2_list.append(v2)
            Omgw_scaled_list.append(Omgw_scaled)
            param_list.append([alpha, vw])
            p_cwg_list.append(p_cwg)
            p_ssm_list.append(p_ssm)

    # Get the number of CPUs available to the current process
    # https://stackoverflow.com/a/55423170/
    # n_cpus = len(os.sched_getaffinity(0))
    # logger.debug(f"Parallelising do_all_plot_ps_compare_nuc across {n_cpus} CPU cores.")

    # futures = []
    # with fut.ThreadPoolExecutor(max_workers=n_cpus) as e:
    #     for vw_list, alpha, in zip(VW_LIST_ALL, const.ALPHA_LIST_ALL):
    #         for vw in vw_list:
    #             future = e.submit(plot_ps_compare_nuc, vw, alpha, save_id, graph_file_type)
    #             futures.append(future)
    #             param_list.append([alpha, vw])
    #
    #     for future in futures:
    #         v2, Omgw_scaled, p_cwg, p_ssm = future.result()
    #         v2_list.append(v2)
    #         Omgw_scaled_list.append(Omgw_scaled)
    #
    #         p_cwg_list.append(p_cwg)
    #         p_ssm_list.append(p_ssm)

    return param_list, v2_list, Omgw_scaled_list, p_cwg_list, p_ssm_list


def do_all_plot_ps_1bubble(save_id=None, graph_file_type=None, debug: bool = False):
    vw_weak_list = [0.92, 0.56, 0.44]
    vw_inter_list = [0.92, 0.56, 0.44]

    alpha_weak = 0.0046
    alpha_inter = 0.050

    vw_list_all = [vw_weak_list, vw_inter_list]
    alpha_list_all = [alpha_weak, alpha_inter]

    f_list = []
    data_lst = []
    for vw_list, alpha, in zip(vw_list_all, alpha_list_all):
        for vw in vw_list:
            if debug:
                f, data = plot_ps_1bubble(vw, alpha, save_id, graph_file_type, debug=debug)
                data_lst.append(data)
            f = plot_ps_1bubble(vw, alpha, save_id, graph_file_type)
            f_list.append(f)

    if debug:
        return f_list, np.array(data_lst)
    return f_list
