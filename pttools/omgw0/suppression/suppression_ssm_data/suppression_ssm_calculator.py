"""
Created on Wed 4/8/2021

@author: cg411

Calculates the Kinetic energy suppression factor for a given set of simulation data
"""
import logging
import os.path
import typing as tp

import numpy as np

import pttools.bubble as bbl
import pttools.ssmtools.const as ssm_const
import pttools.ssmtools.spectrum as ssm

logger = logging.getLogger(__name__)

SUPPRESSION_FOLDER = os.path.dirname(os.path.abspath(__file__))


def calc_sup_ssm(path: str, save: bool = True, npt: ssm_const.NptType = ssm_const.NPTDEFAULT) -> tp.Dict[str, tp.Union[np.ndarray, tp.List[float]]]:
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
    sim_data: np.ndarray = np.loadtxt(path, skiprows=1)
    # Calculating the suppression factor for the SSM
    out_ssm = []
    out_ssm_tot = []
    Ubarf_2_ssm = []
    sup_ssm_all = []

    z = np.logspace(0, 3, 100)

    for i, vw in enumerate(sim_data[:, 0]):
        # print(vw)
        alpha = sim_data[i, 1]
        params = (vw, alpha, ssm.NucType.EXPONENTIAL,(1,))
        out_ssm.append(ssm.power_gw_scaled_bag(z, params, npt=npt))  # omgw_ssm /(HnR*)(Hnt) ,:TODO check how to add these in new PTtools / are they still needed z_st_thresh=np.inf ,npt=[7000,200,1000]
        out_ssm_tot.append(np.trapezoid(out_ssm[i], np.log(z)))
        Ubarf_2_ssm.append(bbl.get_ubarf2_bag(vw, alpha))

        sim_omgw = sim_data[i, 3]  # omgw_sim_tot /(HnR*)(Hnt)
        expected_Ubarf2 = sim_data[i, 5]**2  # Ubarf_exp^2
        sup_ssm = (Ubarf_2_ssm[i]/expected_Ubarf2)**2 * sim_omgw/out_ssm_tot[i]
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
        # print(x)
        new_filename = f"{x[0]}_ssm"
        np.savez(new_filename, **ssm_sup_data)
        # print("SSM suppression file ", new_filename, " created")

    return ssm_sup_data


if __name__ == "__main__":
    calc_sup_ssm("suppression_2.txt")
    calc_sup_ssm("suppression_no_hybrids.txt")
