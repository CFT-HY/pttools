"""Constants common to both ssm_compare.py and ssm_paper_utils.py"""

import pttools.ssmtools as ssm

ALPHA_WEAK = 0.0046
ALPHA_INTER = 0.050
ALPHA_LIST_ALL = [ALPHA_WEAK, ALPHA_INTER]

COLOURS = ("b", "r", "g")

ETA_WEAK_LIST = [0.19, 0.35, 0.51, 0.59, 0.93]
ETA_INTER_LIST = [0.17, 0.40, 0.62]

FILE_TYPE = "pdf"

# Number of points used in the numerical calculations (n_z, n_xi, n_t)
# z - wavenumber space, xi - r/t space, t - time for size distribution integration
NP_LIST = [[1000, 2000, 200], [2500, 5000, 500], [5000, 10000, 1000]]
# Additional values from ssm_paper_utils.py
# Np_list = [[2000, 2000, 200], ]
# Np_list = [[1000, 1000, 200], [2000, 2000, 200], [5000, 5000, 200]]
# Np_list = [[10000, 10000, 200],]

NUC_TYPE = ssm.NucType.SIMULTANEOUS
NUC_ARGS = (1.,)
# Simultaneous is relevant for comparison to num sims
# Or: (ssm_compare.py)
# NUC_TYPE = "exponential"
# NUC_ARGS = (1,)
# Or: (ssm_paper_utils.py)
# NUC_TYPE = "exponential"
# NUC_ARGS = (0,)

NZ_STRING = "nz" + "".join(f"{np[0] // 1000}k" for np in NP_LIST)
NUC_STRING = NUC_TYPE[0:3] + "_" + "_".join(str(arg) for arg in NUC_ARGS) + "_"

NT_STRING = f"_nT{NP_LIST[0][2]}"

VW_WEAK_LIST = [0.92, 0.80, 0.68, 0.56, 0.44]

Z_MIN = 0.2   # Minimum z = k.R* array value
Z_MAX = 1000  # Maximum z = k.R* array value
