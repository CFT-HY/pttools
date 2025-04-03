"""
Created on Wed 4/8/2021

@author: cg411

Removes the unpublished hybrid data from the suppression data set
"""

import logging
import os.path

import numpy as np
import pttools.bubble as bbl

logger = logging.getLogger(__name__)

SUPPRESSION_FOLDER = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(SUPPRESSION_FOLDER, "suppression_2.txt")


def remove_hybrids(path: str = DEFAULT_PATH, suffix: str = "") -> str:
    """
    Removing hybrids from simulation data
    order of entries in txt file
    vw alph suppress sim_omgw exp_omgw exp_ubarf
    """
    sim_data: np.ndarray = np.loadtxt(path, skiprows=1)

    vw_no_hybrid = []
    al_no_hybrid = []
    sup_sim_no_hybrids = []
    sim_omgw_no_hybrids = []
    exp_omgw_no_hybrids = []
    exp_Ubarf_no_hybrids = []

    # speed of sound
    cs = 1/np.sqrt(3)

    for i, vw in enumerate(sim_data[:, 0]):
        alpha = sim_data[i, 1]

        if cs < vw < bbl.v_chapman_jouguet_bag(alpha):
            # logger.debug("Ignoring hybrid for i=%s, vw=%s", i, vw)
            pass
        else:
            vw_no_hybrid.append(sim_data[i, 0])
            al_no_hybrid.append(sim_data[i, 1])
            sup_sim_no_hybrids .append(sim_data[i, 2])
            sim_omgw_no_hybrids.append(sim_data[i, 3])
            exp_omgw_no_hybrids.append(sim_data[i, 4])
            exp_Ubarf_no_hybrids.append(sim_data[i, 5])

    out_path = os.path.join(SUPPRESSION_FOLDER, f"suppression_no_hybrids{f'_{suffix}' if suffix else ''}.txt")
    with open(out_path, 'w') as f:
        f.write("vw" + " " + "alph" + " " + "suppress" + " " + "sim_omgw" + " " + "exp_omgw" + "exp_ubarf" )
        f.write('\n')

        for i in range(0, len(vw_no_hybrid)):
            line = str(vw_no_hybrid[i]) + " " + str(al_no_hybrid[i]) + " " + str(sup_sim_no_hybrids[i]) + " " + str(sim_omgw_no_hybrids[i]) + " " + str(exp_omgw_no_hybrids[i]) + " " + str(exp_Ubarf_no_hybrids[i])
            f.write(line)
            f.write('\n')

    logger.debug("Simulation suppression data without hybrids file created.")
    return out_path


if __name__ == "__main__":
    remove_hybrids()
