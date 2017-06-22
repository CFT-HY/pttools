# A program to generate plots of v and ln(T/T_c) against xi.
# May in future split into two files, one containing the program itself, and a toolbox-type file.

# To do:
# Try to identify xi_end (see findTminus) mathematically rather than guessing
# -> Find expression for cs minus (Espinosa pg 10)
# 0<alphaPlus<1/3
import numpy as np
import EIKR_Toolbox as eikr


def min_speed_deton(al_p, c):
    # Minimum speed for a detonation
    return (c/(1 + al_p))*(1 + np.sqrt(al_p*(2. + 3.*al_p)))


# def max_speed_deflag(al_p, c):
#     # Maximum speed for a deflagration
#     vm=cs
#     return 1/(3*vPlus(vm, al_p, 'Deflagration'))


def IdentifyType(vw, al_p, c):
    # vw = wall velocity, al_p is alpha plus, cs is speed of sound (varies dependent and EoS used).
    if vw < c:
        wallType = 'Def'
        vm = vw
        vp = vPlus(vm, al_p)
    elif vw > c:
        if vw < min_speed_deton(al_p, c):
            wallType = 'Hyb'
            vm = c
            vp = vPlus(vm, al_p)
        else:
            wallType = 'Det'
            vp = vw
            vm = vMinus(vp, al_p)
    # Should consider case where vWall==cs
    print 'Using wall type ', wallType
    print 'v- = ', vm
    print 'v+  = ', vp
    return wallType, vp, vm


def vPlus(vm, al_p):
    if vm < 1./np.sqrt(3.):
        vp = (1./(1.+al_p))*(((vm/2.)+(1./6.*vm))-np.sqrt(((vm/2.)+(1./6.*vm))**2+(al_p**2)+((2./3.)*al_p)-(1./3.)))
    elif vm > 1./np.sqrt(3.):
        vp = (1./(1.+al_p))*(((vm/2.)+(1./6.*vm))+np.sqrt(((vm/2.)+(1./6.*vm))**2+(al_p**2)+((2./3.)*al_p)-(1./3.)))
    # Edge case?
    return vp


def vMinus(vp, al_p):
    if vp < 1./np.sqrt(3.):
        vm = (((1+al_p)/2.)*vp+((1.-3.*al_p)/6*vp))-np.sqrt((((1.+al_p)/2.)*vp+((1.-3.*al_p)/6.*vp))**2-(1./3.))
    elif vp > 1./np.sqrt(3.):
        vm = (((1+al_p)/2.)*vp+((1.-3.*al_p)/6*vp))+np.sqrt((((1.+al_p)/2.)*vp+((1.-3.*al_p)/6.*vp))**2-(1./3.))
    # Edge case?
    return vm

# def xi_stop(wallType, vw):
#     if wallType == 'Def':
#         xs = vw # wrong side, want xi shock
#     else:
#         xs = csMinus() # T dep, don't know final T
#     return xs


def cs(T, state):
    if state == 'EIKR':
        return eikr.cs(T)
    else:
        return 1./np.sqrt(3.0)
