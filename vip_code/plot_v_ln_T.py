# A program to generate plots of v and ln(T/T_c) against xi.
# May in future split into two files, one containing the program itself, and a toolbox-type file.

# To do:
# Try to identify xi_end (see findTminus) mathematically rather than guessing
# -> Find expression for cs minus (Espinosa pg 10)
# 0<alphaPlus<1/3
import numpy as np
from scipy import integrate as itg
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


def mu(a, b):
    return (a-b)/(1.-a*b)


def dv_dxi_deflag(vp, x, c):
    # differential equation: dv/dxi  for deflgrations
    if vp < vShock(x, c): # Can happen if you try to integrate beyond shock
        val = 0.       # Stops solution blowing up
    else:
        val = (2./x)*vp*(1.-vp**2)*(1./(1-x*vp))*(1./(mu(x,vp)**2/c**2 - 1))
    return val


def dv_dxi_deton(vp, x, c):
    #    differential equation: dv/dxi  for detonations and hybrids (integrate backwards from wall)
    val = (2./x)*vp*(1.-vp**2)*(1./(1-x*vp))*(1./(mu(x,vp)**2/c**2 - 1))
    return val


def vShock(xis, c):
    # Fluid velocity at a shock at xis.  No shock for xis < cs, so returns zero
    return np.maximum(0., (xis**2-c**2)/(xis*(1-c**2)))


def xvariables(Npts, v_w):
    dxi = 1./Npts
    xi = np.linspace(0., 1., num=(Npts+1))
    v_sh = np.zeros(Npts+1)
    nWall = np.int(np.floor(v_w/dxi))
    ncs = np.int(np.floor(cs/dxi))
    v_sh[ncs:] = vShock(xi[ncs:])
    xi_ahead = xi[nWall:]
    xi_behind = xi[nWall-1:ncs:-1]
    return dxi, xi, v_sh, nWall, ncs, xi_ahead, xi_behind


def v_just_behind(x,v,dx):
    # Fluid velocity one extra space step behind wall, arranged so that dv_dxi_deton guaranteed positive
    dv = np.sqrt(4.*dx*v*(1-v*v)*(x-v)/(1-x*x))
    return v-dv


def velocity(vw, vp, vm, wallType, Npts):
    vFluid = np.zeros([Npts + 1, 1])  # Could make size [Npts,len(v_w)]
    dxi, xi, v_sh, nWall, ncs, xi_ahead, xi_behind = xvariables(Npts, vw)  # initiating x-axis variables
    vpPrime = mu(vw, vp)
    vmPrime = mu(vw, vm)
    #   Calculating the fluid velocity
    vFluid[nWall:] = itg.odeint(dv_dxi_deflag, vpPrime, xi_ahead)
    vFluid[nWall - 1:ncs:-1] = itg.odeint(dv_dxi_deton, v_just_behind(xi[nWall], vmPrime, dxi), xi_behind)
    if not (wallType == "Det"):
        for n in range(nWall, Npts):
            if vFluid[n] < v_sh[n]:
                nShock = n
                break
    else:
        nShock = nWall

    # Set fluid velocity to zero in front of the shock (integration isn't correct in front of shock)
    vFluid[nShock:] = 0.0

    return vFluid


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
