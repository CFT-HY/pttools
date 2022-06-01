"""
Constants common to both
:mod:`tests.paper.ssm_compare`
and
:mod:`tests.paper.ssm_paper_utils`.
"""

import numpy as np

import pttools.ssmtools as ssm

#: Weak transition strength
ALPHA_WEAK = 0.0046
#: Intermediate transition strength
ALPHA_INTER = 0.050
#: Transition strengths to plot with
ALPHA_LIST_ALL = [ALPHA_WEAK, ALPHA_INTER]

#: Colours for comparison plots
COLOURS = ("b", "r", "g")

ETA_WEAK_LIST = [0.19, 0.35, 0.51, 0.59, 0.93]
ETA_INTER_LIST = [0.17, 0.40, 0.62]

#: File type for saving the plots
FILE_TYPE = "pdf"

#: Number of points used in the numerical calculations (n_z, n_xi, n_t).
#: z - wavenumber space, xi - r/t space, t - time for size distribution integration
NP_ARR = np.array([[1000, 2000, 200], [2500, 5000, 500], [5000, 10000, 1000]])
# Additional values from ssm_paper_utils.py
# NP_ARR = [[2000, 2000, 200], ]
# NP_ARR = [[1000, 1000, 200], [2000, 2000, 200], [5000, 5000, 200]]
# NP_ARR = [[10000, 10000, 200],]

#: Default nucleation type
NUC_TYPE = ssm.NucType.SIMULTANEOUS
#: Default nucleation arguments
NUC_ARGS = (1.,)
# Simultaneous is relevant for comparison to num sims
# Or: (ssm_compare.py)
# NUC_TYPE = "exponential"
# NUC_ARGS = (1,)
# Or: (ssm_paper_utils.py)
# NUC_TYPE = "exponential"
# NUC_ARGS = (0,)

NZ_STRING = "nz" + "".join(f"{np[0] // 1000}k" for np in NP_ARR)
NUC_STRING = NUC_TYPE[0:3] + "_" + "_".join(str(arg) for arg in NUC_ARGS) + "_"
#: Nucleation config as string
NUC_STRING: str = NUC_TYPE[0:3] + "_" + "_".join(str(arg) for arg in NUC_ARGS) + "_"

NT_STRING = f"_nT{NP_ARR[0][2]}"

#: Wall velocities for testing with weak transition strength
VW_WEAK_LIST = [0.92, 0.80, 0.68, 0.56, 0.44]

Z_MIN = 0.2   # Minimum z = k.R* array value
Z_MAX = 1000  # Maximum z = k.R* array value
