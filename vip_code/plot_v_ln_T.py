# A program to generate plots of v and ln(T/T_c) against xi.
# May in future split into two files, one containing the program itself, and a toolbox-type file.

# To do:
# Try to identify xi_end (see findTminus) mathematically rather than guessing
# -> Find expression for cs minus (Espinosa pg 10)
import sys
import numpy as np
from scipy import integrate as itg
from scipy import optimize as opt
import EIKR_Toolbox as Eikr
import Bag_Toolbox as Bag

def min_speed_deton(al_p, c):
    # Minimum speed for a detonation
    return (c/(1 + al_p))*(1 + np.sqrt(al_p*(2. + 3.*al_p)))


# def max_speed_deflag(al_p, c):
#     # Maximum speed for a deflagration
#     vm=cs
#     return 1/(3*vPlus(vm, al_p, 'Deflagration'))


def identify_type(vw, al_p, cs):
    # vw = wall velocity, al_p is alpha plus, cs is speed of sound (varies dependent and EoS used).
    if vw < cs:
        walltype = 'Def'
        vm = vw
        vp = vplus(vm, al_p)
    elif vw > cs:
        if vw < min_speed_deton(al_p, cs):
            walltype = 'Hyb'
            vm = cs
            vp = vplus(vm, al_p)
        else:
            walltype = 'Det'
            vp = vw
            vm = vminus(vp, al_p)
    # Should consider case where vWall==cs
    print 'Using wall type ', walltype
    print 'v- = ', vm
    print 'v+  = ', vp
    return walltype, vp, vm


def vplus(vm, al_p):
    if vm < 1./np.sqrt(3.):
        vp = (1./(1.+al_p))*(((vm/2.)+(1./6.*vm))-np.sqrt(((vm/2.)+(1./6.*vm))**2+(al_p**2)+((2./3.)*al_p)-(1./3.)))
    elif vm > 1./np.sqrt(3.):
        vp = (1./(1.+al_p))*(((vm/2.)+(1./6.*vm))+np.sqrt(((vm/2.)+(1./6.*vm))**2+(al_p**2)+((2./3.)*al_p)-(1./3.)))
    # Edge case?
    return vp


def vminus(vp, al_p):
    if vp < 1./np.sqrt(3.):
        vm = (((1+al_p)/2.)*vp+((1.-3.*al_p)/6*vp))-np.sqrt((((1.+al_p)/2.)*vp+((1.-3.*al_p)/6.*vp))**2-(1./3.))
    elif vp > 1./np.sqrt(3.):
        vm = (((1+al_p)/2.)*vp+((1.-3.*al_p)/6*vp))+np.sqrt((((1.+al_p)/2.)*vp+((1.-3.*al_p)/6.*vp))**2-(1./3.))
    # Edge case?
    return vm


def mu(a, b):
    return (a-b)/(1.-a*b)


def dv_dxi_deflag(vp, x, cs):
    # differential equation: dv/dxi  for deflagrations
    if vp < v_shock(x, cs):  # Can happen if you try to integrate beyond shock
        val = 0.  # Stops solution blowing up
    else:
        val = (2./x)*vp*(1.-vp**2)*(1./(1-x*vp))*(1./(mu(x, vp)**2/cs**2 - 1))
    return val


def dv_dxi_deton(vp, x, cs):
    # differential equation: dv/dxi  for detonations and hybrids (integrate backwards from wall)
    val = (2./x)*vp*(1.-vp**2)*(1./(1-x*vp))*(1./(mu(x, vp)**2/cs**2 - 1))
    return val


def v_shock(xis, cs):
    # Fluid velocity at a shock at xis.  No shock for xis < cs, so returns zero
    return np.maximum(0., (xis**2-cs**2)/(xis*(1-cs**2)))


# def xi_stop(wallType, vw):
#     if wallType == 'Def':
#         xs = vw # wrong side, want xi shock
#     else:
#         xs = csMinus() # T dep, don't know final T
#     return xs


def cs_select(t, state):
    if state == 'eikr':
        return Eikr.cs(t)
    else:
        return 1./np.sqrt(3.0)


def get_t_minus(tplus, xiw, eos, vp, vm, ap, am):
    if eos == 'eikr':
        tm = opt.fsolve(Eikr.delta_w, tplus, xiw)
    else:
        tm = Bag.t_minus(tplus, vp, vm, ap, am)
    return tm


def shell_prop(vwall, al_p, tp, ):
    stuff


def main():
    if not len(sys.argv) in [3, 4]:
        sys.stderr.write('usage: %s <v_wall> <secondary_parameter> <state_eqn> [npts]\n' % sys.argv[0])
        sys.exit(1)

    # Error-check input parameters
    state_eqn = sys.argv[3]
    if not state_eqn == 'bag' or not state_eqn == 'eikr':
        while not state_eqn == 'bag' or not state_eqn == 'eikr':
            print 'Error: Unrecognised input'
            state_eqn = input('Enter equation of state model: bag or eikr ')

    v_wall = float(sys.argv[1])
    if not 0. < v_wall < 1.:
        while not 0 < v_wall < 1:
            print 'Error: v_wall must be satisfy 0 < v_wall < 1'
            v_wall = input('Enter v_wall ')

    if len(sys.argv) == 4:
        npts = int(sys.argv[4])
    else:
        npts = 5000
        print 'npts unspecified, defaulting to npts = 5000'

    # select meaning of secondary_parameter based on model used
    if state_eqn == 'bag':
        alpha_plus = float(sys.argv[2])
        if not 0. < alpha_plus < 1./3.:
            while not 0 < alpha_plus < 1./3.:
                print 'Error: alpha_plus must be satisfy 0 < alpha_plus < 1/3'
                alpha_plus = input('Enter alpha_plus')
        # machinery for t_plus(al_p)
    else:
        t_plus = float(sys.argv[2])
        # error check
        alpha_plus = Eikr.alphaplus(t_plus)

    c_s = cs_select(t_plus, state_eqn)
    wall_type, v_plus, v_minus = identify_type(v_wall, alpha_plus, c_s)
    t_minus = get_t_minus(t_plus, v_wall, state_eqn, v_plus, v_minus, a_plus, a_minus)


if __name__ == '__main__':
    main()
