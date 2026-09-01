"""Compute the kinetic energy suppression factor for a given set of simulation data"""

import logging
import os.path

import numpy as np

from pttools.bubble import \
    CS2_BAG_SCALAR_PTR, DEFAULT_ADIABATIC_INDEX, DF_DTAU_PTR_BAG, get_ubarf2_bag
from pttools.ssm.const import NptType, DEFAULT_N_PT
from pttools.ssm.spectrum import NucType
from pttools.ssm.spectrum_bag import power_gw_bag
import pttools.type_hints as th

logger = logging.getLogger(__name__)

SUPPRESSION_FOLDER: str = os.path.dirname(os.path.abspath(__file__))


def calc_sup_ssm(
        path: str,
        save: bool = True,
        npt: NptType = DEFAULT_N_PT,
        lambda_correction: bool = False) -> dict[str, th.FloatArr1DOrList]:
    """
    file must be a txt file with data in columns as follows
    vw alpha suppression_sim sim_omgw exp_omgw exp_ubarf
    where vw = wall speed
    alpha = transition strength
    suppression_sim
    sim_omgw = total (integrated (omgw_ssm /(HnR*)(Hnt)) )
    exp_omgw = same as above but expected quantity
    exp_ubarf = expected quantity for ubarf
    """
    if not os.path.isabs(path):
        path = os.path.join(SUPPRESSION_FOLDER, path)
    sim_data = np.loadtxt(path, skiprows=1)

    out_ssm_tot = []
    Ubarf_2_ssm = []
    sup_ssm_all = []

    z = np.logspace(0, 3, 100)

    for i, vw in enumerate(sim_data[:, 0]):
        alpha = sim_data[i, 1]
        sim_omgw = sim_data[i, 3]  # omgw_sim_tot /(HnR*)(Hnt)
        expected_Ubarf2 = sim_data[i, 5] ** 2  # Ubarf_exp^2

        # TODO: Check how to add these in new PTtools / are they still needed: z_st_thresh=np.inf, npt=[7000,200,1000]
        out_ssm = 3 * DEFAULT_ADIABATIC_INDEX ** 2 * power_gw_bag(
            z=z,
            params=(vw, alpha, NucType.EXPONENTIAL,(1,)),
            npt=npt,
            lambda_correction=lambda_correction
        )  # omgw_ssm /(HnR*)(Hnt)
        out_ssm_tot.append(np.trapezoid(out_ssm, np.log(z)))
        Ubarf_2_ssm.append(get_ubarf2_bag(
            vw, alpha, cs2_fun_ptr=CS2_BAG_SCALAR_PTR, df_dtau_ptr=DF_DTAU_PTR_BAG))

        sup_ssm = (Ubarf_2_ssm[i] / expected_Ubarf2)**2 * sim_omgw / out_ssm_tot[i]
        sup_ssm_all.append(sup_ssm)

    ssm_sup_data = {
        "vw_sim": sim_data[:, 0],
        "alpha_sim": sim_data[:, 1],
        "sup_ssm": sup_ssm_all,
        "Ubarf_2_ssm": Ubarf_2_ssm,
        "ssm_tot": out_ssm_tot
    }

    if save:
        x = path.split(".txt")
        np.savez(f"{x[0]}_ssm", **ssm_sup_data)

    return ssm_sup_data


if __name__ == "__main__":
    calc_sup_ssm("suppression_2.txt")
    calc_sup_ssm("suppression_no_hybrids.txt")
