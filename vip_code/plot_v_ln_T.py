# A program to generate plots of v and ln(T/T_c) against xi.
# May in future split into two files, one containing the program itself, and a toolbox-type file.

# To do:
# Try to identify xi_end (see findTminus) mathematically rather than guessing
# -> Find expression for cs minus (Espinosa pg 10)

import numpy as np
from scipy import integrate as itg
from scipy import optimize as opt
import EIKR_Toolbox as Eikr


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
        wall_type = 'Def'
        vm = vw
        vp = v_plus(vm, al_p)
    elif vw > cs:
        if vw < min_speed_deton(al_p, cs):
            wall_type = 'Hyb'
            vm = cs
            vp = v_plus(vm, al_p)
        else:
            wall_type = 'Det'
            vp = vw
            vm = v_minus(vp, al_p)
    # Should consider case where vWall==cs
    print 'Using wall type ', wall_type
    print 'v- = ', vm
    print 'v+  = ', vp
    return wall_type, vp, vm


def v_plus(vm, al_p):
    if vm < 1./np.sqrt(3.):
        vp = (1./(1.+al_p))*(((vm/2.)+(1./6.*vm))-np.sqrt(((vm/2.)+(1./6.*vm))**2+(al_p**2)+((2./3.)*al_p)-(1./3.)))
    elif vm > 1./np.sqrt(3.):
        vp = (1./(1.+al_p))*(((vm/2.)+(1./6.*vm))+np.sqrt(((vm/2.)+(1./6.*vm))**2+(al_p**2)+((2./3.)*al_p)-(1./3.)))
    # Edge case?
    return vp


def v_minus(vp, al_p):
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


def x_variables(npts, v_w, cs):
    dxi = 1./npts
    xi = np.linspace(0., 1., num=(npts+1))
    v_sh = np.zeros(npts+1)
    n_wall = np.int(np.floor(v_w/dxi))
    ncs = np.int(np.floor(cs/dxi))
    v_sh[ncs:] = v_shock(xi[ncs:], cs)
    xi_ahead = xi[n_wall:]
    xi_behind = xi[n_wall-1:ncs:-1]
    return dxi, xi, v_sh, n_wall, ncs, xi_ahead, xi_behind


def v_just_behind(x, v, dx):
    # Fluid velocity one extra space step behind wall, arranged so that dv_dxi_deton guaranteed positive
    dv = np.sqrt(4.*dx*v*(1-v*v)*(x-v)/(1-x*x))
    return v-dv


def velocity(vw, vp, vm, wall_type, cs, npts):
    v_fluid = np.zeros([npts+1, 1])  # Could make size [npts,len(v_w)]
    dxi, xi, v_sh, n_wall, ncs, xi_ahead, xi_behind = x_variables(npts, vw, cs)  # initiating x-axis variables
    vp_prime = mu(vw, vp)
    vm_prime = mu(vw, vm)
    #   Calculating the fluid velocity
    v_fluid[n_wall:] = itg.odeint(dv_dxi_deflag, vp_prime, xi_ahead)
    v_fluid[n_wall - 1:ncs:-1] = itg.odeint(dv_dxi_deton, v_just_behind(xi[n_wall], vm_prime, dxi), xi_behind)
    if not (wall_type == "Det"):
        for n in range(n_wall, npts):
            if v_fluid[n] < v_sh[n]:
                n_shock = n
                break
    else:
        n_shock = n_wall

    # Set fluid velocity to zero in front of the shock (integration isn't correct in front of shock)
    v_fluid[n_shock:] = 0.0

    return v_fluid


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


def get_Tminus(Tguess, Xiw):
    Tm = opt.fsolve(Eikr.delta_w, Tguess, Xiw)
    return Tm


def main():
    # Give input parameters
    state_eqn = input('Enter equation of state model: bag or eikr ')
    if not state_eqn == 'bag' or not state_eqn == 'eikr':
        while not state_eqn == 'bag' or not state_eqn == 'eikr':
            print 'Error: Unrecognised input'
            state_eqn = input('Enter equation of state model: bag or eikr ')
    v_wall = input('Enter v_wall ')
    if not 0 < v_wall < 1:
        while not 0 < v_wall < 1:
            print 'Error: v_wall must be satisfy 0<v_wall<1'
            v_wall = input('Enter v_wall ')
    point_success = 0
    while point_success == 0:
        points = raw_input('Enter number of points to integrate over. Leave blank to default to 5000 ') or 5000
        try:
            points = int(points)
            point_success = 1
        except ValueError:
            print 'Error: Please enter an integer'

    if state_eqn == 'eikr':
        tg = 0.8
        t_minus = get_Tminus(tg, v_wall)

    # select method of defining alpha_plus based on model used
    if state_eqn == 'bag':
        alpha_plus_e = input('Enter alpha_plus')
        if not 0 < alpha_plus_e < 1./3.:
            while not 0 < alpha_plus_e < 1./3.:
                print 'Error: alpha_plus must be satisfy 0<v_wall<1/3'
                alpha_plus_e = input('Enter alpha_plus')
        alpha_plus = np.full((points), alpha_plus_e)
    else:
        alpha_plus = Eikr.alphaplus(t)



