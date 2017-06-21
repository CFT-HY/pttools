# A program to generate plots of v and ln(T/T_c) against xi.
# May in future split into two files, one containing the program itself, and a toolbox-type file.

# To do: vPlus/vMinus functions, find eqn for cs from EIKR EoS, try to identify xi_end (see findTminus) mathematically
# rather than guessing
import numpy as np


def min_speed_deton(al_p, cs):
    # Minimum speed for a detonation
    return (cs/(1 + al_p))*(1 + np.sqrt(al_p*(2. + 3.*al_p)))


# def max_speed_deflag(al_p, cs):
#     # Maximum speed for a deflagration
#     vm=cs
#     return 1/(3*vPlus(vm, al_p, 'Deflagration'))


def IdentifyType(vw, al_p, cs):
    # vw = wall velocity, al_p is alpha plus, cs is speed of sound (varies dependent and EoS used).
    # vPlus and vMinus are functions based on 2016 work, to be clarified with Mark.
    if vw < cs:
        wallType = 'Def'
        vm = vw
        vp = vPlus(vm)
    elif vw > cs:
        if vw < min_speed_deton(al_p, cs):
            wallType = 'Hyb'
        else:
            wallType = 'Det'
            vp = vw
            vm = vMinus(vp)
    # Should consider case where vWall==cs
    print 'Using wall type ', wallType
    print 'v- = ', vm
    print 'v+  = ', vp
    return wallType, vp, vm
