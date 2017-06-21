# A program to generate plots of v and ln(T/T_c) against xi.
# May in future split into two files, one containing the program itself, and a toolbox-type file.

import numpy as np


def min_speed_deton(al_p, cs):
    # Minimum speed for a detonation
    return (cs/(1 + al_p))*(1 + np.sqrt(al_p*(2. + 3.*al_p)))


# def max_speed_deflag(al_p, cs):
#     # Maximum speed for a deflagration
#     vm=cs
#     return 1/(3*vPlus(vm, al_p, 'Deflagration'))


def IdentifyType(vWall, al_p, cs):
    if vWall<cs:
        wallType = 'Def'
    elif vWall>cs:
        if vWall<min_speed_deton(al_p, cs):
            wallType = 'Hyb'
        else:
            wallType = 'Det'
