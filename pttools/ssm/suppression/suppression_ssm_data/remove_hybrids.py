"""Remove the unpublished hybrid data from the suppression data set."""

import logging
import os
from pathlib import Path

import numpy as np

from pttools.bubble import CS0, v_chapman_jouguet_bag

logger: logging.Logger = logging.getLogger(__name__)

SUPPRESSION_FOLDER: Path = Path(__file__).resolve().parent
DEFAULT_PATH: Path = SUPPRESSION_FOLDER / "suppression_2.txt"


def remove_hybrids(path: str | os.PathLike[str] = DEFAULT_PATH, suffix: str = "") -> Path:
    """
    Removing hybrids from simulation data.

    The order of entries in txt file should be:
    vw alph suppress sim_omgw exp_omgw exp_ubarf
    """
    sim_data = np.loadtxt(path, skiprows=1)

    vw_no_hybrid = []
    al_no_hybrid = []
    sup_sim_no_hybrids = []
    sim_omgw_no_hybrids = []
    exp_omgw_no_hybrids = []
    exp_Ubarf_no_hybrids = []

    for i, vw in enumerate(sim_data[:, 0]):
        alpha = sim_data[i, 1]

        if CS0 < vw < v_chapman_jouguet_bag(alpha):
            # logger.debug("Ignoring hybrid for i=%s, vw=%s", i, vw)
            pass
        else:
            vw_no_hybrid.append(sim_data[i, 0])
            al_no_hybrid.append(sim_data[i, 1])
            sup_sim_no_hybrids .append(sim_data[i, 2])
            sim_omgw_no_hybrids.append(sim_data[i, 3])
            exp_omgw_no_hybrids.append(sim_data[i, 4])
            exp_Ubarf_no_hybrids.append(sim_data[i, 5])

    out_path = SUPPRESSION_FOLDER / f"suppression_no_hybrids{f'_{suffix}' if suffix else ''}.txt"
    with out_path.open("w") as f:
        f.write("vw" + " " + "alph" + " " + "suppress" + " " + "sim_omgw" + " " + "exp_omgw" + "exp_ubarf" )
        f.write('\n')

        for i in range(len(vw_no_hybrid)):
            line = (
                str(vw_no_hybrid[i]) + " " + str(al_no_hybrid[i]) + " " + str(sup_sim_no_hybrids[i]) + " "
                + str(sim_omgw_no_hybrids[i]) + " " + str(exp_omgw_no_hybrids[i]) + " "
                + str(exp_Ubarf_no_hybrids[i])
            )
            f.write(line)
            f.write("\n")

    logger.debug("Simulation suppression data without hybrids file created.")
    return out_path


if __name__ == "__main__":
    remove_hybrids()
