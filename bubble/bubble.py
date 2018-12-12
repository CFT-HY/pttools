#!/usr/bin/env python
#
# Functions for calculating fluid profile around expanding Higgs-phase bubble.
# See Espinosa et al 2010
#
# Mark Hindmarsh 2015-18
# with Mudhahir Al-Ajmi and
# contributions from: 
# Danny Bail 2016-18
# Jacky Lindsay and Mike Soughton 2017-18
#
# Planned changes 12.18:
# - allow general equation of state (so integrate with V, T together instead of v, w separately)
#   Idea to introduce eos as a class. Need a new interface which uses eos variables rather than alpha.

from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import scipy.integrate as spi
import scipy.optimize as opt

eps = np.nextafter(0, 1)  # smallest float

# Default number of entries in xi array
NPDEFAULT = 5000
NPMAX = 1000000
# How accurate is alpha_plus(alpha_n)
find_alpha_plus_tol=1e-6
# Integration limit for parametric form of fluid equations
TENDDEFAULT = 50.
dxi_small = 1./NPDEFAULT


# Some functions useful for the bag equation of state.

cs0 = 1/np.sqrt(3)  # ideal speed of sound
cs0_2 = 1./3  # ideal speed of sound squared

#def cs_fun(w):
#    # Speed of sound function
#    # to be adapted to more realistic equations of state, e.g. with interpolation
#    return cs0
#
#
def cs_w(w):
    # Speed of sound function, another label
    # to be adapted to more realistic equations of state, e.g. with interpolation
    return cs0


def cs2_w(w):
    # Speed of sound squared function
    # to be adapted to more realistic equations of state, e.g. with interpolation
    return cs0_2


def cs2_bag(w):
    return 1/3.


def p(w, phase, theta_s, theta_b=0.):
    # pressure as a function of enthalpy, assuming bag model
    # theta = (e - 3p)/4 ("vacuum energy")
    # _s = symmetric phase, ahead of bubble (phase = 0)
    # _b = broken phase, behind bubble (phase = 1)
    # enthalpy, theta and phase can be arrays (same shape)
    theta = theta_b*phase + theta_s*(1.0 - phase)
    return 0.25*w - theta
    

def e(w, phase, theta_s, theta_b=0.):
    # energy density as a function of enthalpy, assuming bag model
    # theta = (e - 3p)/4 ("vacuum energy")
    # _s = symmetric phase, ahead of bubble (phase = 0)
    # _b = broken phase, behind bubble (phase = 1)
    # enthalpy and phase can be arrays (same shape)
    return w - p(w,phase,theta_s,theta_b)


def w(e, phase, theta_s, theta_b=0):
    # e enthalpy as a function of energy density, assuming bag model
    # theta = (e - 3p)/4 ("vacuum energy")
    # _s = symmetric phase, ahead of bubble (phase = 0)
    # _b = broken phase, behind bubble (phase = 1)
    # enthalpy and phase can be arrays (same shape)
    # Actually, theta is often known only from alpha_n and w, so should
    # think about an fsolve?
    theta = theta_b*phase + theta_s*(1.0 - phase)
    return (4/3)*(e - theta)


def phase(xi,v_w):
    # Returns array indicating phase of system is 
    # in symmetric phase (xi>v_w), phase = 0 
    # in broken phase (xi<v_w), phase = 1
    ph = np.zeros_like(xi)
    ph[np.where(xi < v_w)] = 1.0
    return ph

    
# Relativity helper functions

def lorentz(xi, v):
    # Lorentz Transformation of fluid speed v between local moving frame and plasma frame.
    return (xi - v)/(1 - v*xi)


def gamma2(v):
    # Calculate square of Lorentz gamma
    return 1./(1. - v**2)


def gamma(v):
    # Calculate Lorentz gamma
    return np.sqrt(gamma2(v))


# Boundary conditions at bubble wall 

def v_plus(vm, ap, wall_type):
    # Wall frame speed ahead of the wall
    X = vm + 1./(3*vm)
    if wall_type == 'Detonation':
        b = 1.
    else:
        b = -1.
    return_value = (0.5/(1+ap))*(X + b*np.sqrt(X**2 + 4.*ap**2 + (8./3.)*ap - (4./3.)))

    if isinstance(return_value, np.ndarray):
        return_value[np.where(isinstance(return_value,complex))] = np.nan
    else:
        if isinstance(return_value,complex):
            return_value = np.nan
            
    return return_value


def v_minus(vp, ap, wall_type='Detonation'):
    #    Wall frame speed behind the wall
    vp2 = vp**2
    Y = vp2 + 1./3.
    Z = (Y - ap*(1. - vp2))
    X = (4./3.)*vp2

    if wall_type=='Detonation':
        b = +1.
    else:
        b = -1.
    
    return_value = (0.5/vp)*(Z + b*np.sqrt(Z**2 - X))

    if isinstance(return_value, np.ndarray):
        return_value[np.where(isinstance(return_value,complex))] = np.nan
    else:
        if isinstance(return_value,complex):
            return_value = np.nan
            
    return return_value

    
def fluid_speeds_at_wall(v_wall, alpha_p, wall_type):
    # Sets up boundary conditions at the wall
    # Returns fluid speed vf just behind (m) and just ahead (p) of wall,
    # in wall (_w) and plasma (_p) frames.
    if v_wall <= 1:
        # print( "max_speed_deflag(alpha_p)= ", max_speed_deflag(alpha_p))
        #     if v_wall < max_speed_deflag(alpha_p) and v_wall <= cs and alpha_p <= 1/3.:
        if wall_type == 'Deflagration':
            vfm_w = v_wall                             # Fluid velocity just behind the wall in wall frame (v-)
            vfm_p = lorentz(v_wall, vfm_w)             # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_plus(v_wall, alpha_p, wall_type) # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = lorentz(v_wall, vfp_w)             # Fluid velocity just ahead of the wall in plasma frame
        elif wall_type == 'Hybrid':
            vfm_w = cs0                                # Fluid velocity just behind the wall in plasma frame (hybrid)
            vfm_p = lorentz(v_wall, vfm_w)             # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_plus(cs0, alpha_p, wall_type)     # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = lorentz(v_wall, vfp_w)             # Fluid velocity just ahead of the wall in plasma frame
        elif wall_type == 'Detonation':
            vfm_w = v_minus(v_wall, alpha_p)           # Fluid velocity just behind the wall in wall frame (v-)
            vfm_p = lorentz(v_wall, vfm_w)             # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_wall                             # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = lorentz(v_wall, vfp_w)             # Fluid velocity just ahead of the wall in plasma frame
        else:
            sys.stderr.write("fluid_speeds_at_wall: error: wall_type wrong or unset")
            sys.exit(1)
    else:
        sys.stderr.write("fluid_speeds_at_wall: error: v_wall > 1")

    return vfp_w, vfm_w, vfp_p, vfm_p


def enthalpy_ratio(v_m, v_p):
    # ratio of enthalpies behind and ahead of a shock or transition front
    # From momentum conservation
    # w_-/w_+
    return gamma2(v_m)*v_m/(gamma2(v_p)*v_p)


# Fluid differential equations 
# Now in parametric form (Jacky Lindsay and Mike Soughton MPhys project 2017-18)
# RHS is Eq (33) in Espinosa et al (plus dw/dt not written there)
def df_dtau(y, t, cs2_fun=cs2_bag):
    # Returns differentials in parametric form, suitable for odeint
    v  = y[0]
    w  = y[1]
    xi = y[2]
    cs2 = cs2_fun(w)

    dxi_dt = xi * (((xi - v) ** 2) - (cs2) * ((1 - (xi * v)) ** 2))  # dxi/dt
    dv_dt  = 2 * v * (cs2) * (1 - (v ** 2)) * (1 - (xi * v))  # dv/dt
    dw_dt  = ((2 * v * w / (xi * (xi - v))) * dxi_dt + 
              ((w / (1 - v ** 2)) * (((1 - v * xi) / (xi - v)) + ((xi - v) / (1 - xi * v)))) * dv_dt)  # dw/dt

    return [dv_dt, dw_dt, dxi_dt]

    
def fluid_integrate_param(v0, w0, xi0, t_end=TENDDEFAULT, npts=NPDEFAULT, cs2_fun=cs2_bag):
    # Integrates parametric fluid equations from an initial condition
    # Negative t_end integrates forward
    t = np.linspace(0., t_end, npts)
    if isinstance(xi0, np.ndarray):
        soln = spi.odeint(df_dtau, (v0[0], w0[0], xi0[0]), t, args=(cs2_fun, ))
    else:
        soln = spi.odeint(df_dtau, (v0, w0, xi0), t, args=(cs2_fun, ))
    v   = soln[:, 0]
    w   = soln[:, 1]
    xi  = soln[:, 2]

    return v, w, xi, t


# Useful quantities for deciding type of transition

def min_speed_deton(alpha):
    # Minimum speed for a detonation (Jouguet speed)
    # Equivalent to v_plus(cs0,alpha)
    # Note that alpha_plus = alpha_n for detonation
    return (cs0/(1 + alpha))*(1 + np.sqrt(alpha*(2. + 3.*alpha)))


def max_speed_deflag(alpha_p):
    # Maximum speed for a deflagration: speed where wall and shock are coincident
    # May be greater than 1, meaning that hybrids exist for all wall speeds above cs.
    # alpha_plus < 1/3, but alpha_n unbounded above
    return 1/(3*v_plus(cs0, alpha_p, 'Deflagration'))


def identify_wall_type(v_wall, alpha_n):
    # v_wall = wall velocity, alpha_n = relative trace anomaly at nucleation temp outside shell
    wall_type = 'Error' # Default
    if alpha_n < alpha_n_max_detonation(v_wall):
        # Must be detonation
        wall_type = 'Detonation'
    else:
        if alpha_n < alpha_n_max_deflagration(v_wall):
            if v_wall <= cs0:
                wall_type = "Deflagration"
            else:
                wall_type = "Hybrid"

    return wall_type


def identify_wall_type_alpha_plus(v_wall, alpha_p):
    if v_wall < cs0:
        wall_type = 'Deflagration'
    else:
        if alpha_p < alpha_plus_max_detonation(v_wall):
            wall_type = 'Detonation'
            if alpha_p > alpha_plus_min_hybrid(v_wall) and alpha_p < 1/3.:
                sys.stderr.write('identify_wall_type_alpha_plus: warning:\n')
                sys.stderr.write('      Hybrid and Detonation both possible for v_wall = {}, alpha_p = {}\n'.format(v_wall,alpha_p))
                sys.stderr.write('      Choosing detonation.\n')
        else:
            wall_type = 'Hybrid'


    if alpha_p > (1/3.) and not wall_type == 'Detonation':
        sys.stderr.write('identify_wall_type_alpha_plus: error:\n')
        sys.stderr.write('      no solution for v_wall = {}, alpha_p = {}\n'.format(v_wall,alpha_p))
        wall_type = 'Error'

    return wall_type


# Useful functions for finding properties of solution 

def find_v_index(xi, v_target):
    # Finds array index of xi just above v_target
    n = 0
    it = np.nditer(xi, flags=['c_index'])
    for x in it:
        if x >= v_target:
            n = it.index
            break
    return n


def v_shock(xi):
    # Fluid velocity at a shock at xi.  No shock for xi < cs, so returns zero
    v_sh = (3*xi**2 - 1)/(2*xi)

    if isinstance(v_sh, np.ndarray):
        v_sh[np.where(xi < cs0)] = 0.0
    else:
        if xi < cs0:
            v_sh = 0.0

    return v_sh


def w_shock(xi, w_n=1.):
    # Fluid enthalpy at a shock at xi.  No shock for xi < cs, so returns nan
    w_sh = w_n * (9*xi**2 - 1)/(3*(1-xi**2))

    if isinstance(w_sh, np.ndarray):
        w_sh[np.where(xi < cs0)] = np.nan
    else:
        if xi < cs0:
            w_sh = np.nan

    return w_sh


def find_shock_index(v_f, xi, v_wall, wall_type):
    # Finds array index of shock from where fluid velocity goes below v_shock
    # For detonation, returns wall position.
    n_shock = 0

    if not (wall_type == "Detonation"):
        it = np.nditer([v_f, xi], flags=['c_index'])
        for v, x in it:
            if x > v_wall:
                if v <= v_shock(x):
                    n_shock = it.index
                    break
    else:
        n_shock = find_v_index(xi, v_wall)

    return n_shock


def shock_zoom_last_element(v, w, xi):
# Replaces last element of arrays by better estimate of shock position 
    v_sh = v_shock(xi)
    # First check if last two elements straddle shock
    if v[-1] < v_sh[-1] and v[-2] > v_sh[-2] and xi[-1] > xi[-2]:
        dxi = xi[-1] - xi[-2]
        dv = v[-1] - v[-2]
        dv_sh = v_sh[-1] - v_sh[-2]
        dw_sh = w[-1] - w[-2]
        dxi_sh = dxi * (v[-2] - v_sh[-2])/(dv_sh - dv)
        # now replace final element
        xi[-1] = xi[-2] + dxi_sh
        v[-1]  = v[-2] + (dv_sh/dxi)*dxi_sh
        w[-1]  = w[-2] + (dw_sh/dxi)*dxi_sh
    # If not, do nothing
    return v, w, xi
    

# Main function for integrating fluid equations and deriving v, w
# for complete range 0 < xi < 1

def fluid_shell(v_wall, alpha_n, npts=NPDEFAULT):
    # Finds fluid shell from a given v_wall, alpha_n 
    # v_wall and alpha_plus must be scalars
    wall_type = identify_wall_type(v_wall, alpha_n)
    if wall_type == 'Error':
        sys.stderr.write('fluid_shell: giving up because of identify_wall_type error')
        return np.nan, np.nan, np.nan
    else:
        al_p = find_alpha_plus(v_wall, alpha_n, npts)
        if not np.isnan(al_p):
            return fluid_shell_alpha_plus(v_wall, al_p, wall_type, npts)
        else:
            return np.nan, np.nan, np.nan
        

def fluid_shell_alpha_plus(v_wall, alpha_plus, wall_type='Calculate', npts=NPDEFAULT, w_n=1, cs2_fun=cs2_bag):
    # Integrates fluid equations away from wall.
    # Where v=0 (behind and ahead of shell) uses only two points.
    # v_wall and alpha_plus must be scalars, and are converted from arrays if needed.
    dxi = 1./npts
#    dxi = 10*eps

    if isinstance(alpha_plus, np.ndarray):
        al_p = np.asscalar(alpha_plus)
    else:
        al_p = alpha_plus
    if isinstance(v_wall, np.ndarray):
        v_w = np.asscalar(v_wall)
    else:
        v_w = v_wall

    if wall_type == "Calculate":
        wall_type = identify_wall_type_alpha_plus(v_w, al_p)

    if wall_type == 'Error':
        sys.stderr.write('fluid_shell_alpha_plus: giving up because of identify_wall_type error')
        return np.nan, np.nan, np.nan

    # Solve boundary conditions at wall
    vfp_w, vfm_w, vfp_p, vfm_p = fluid_speeds_at_wall(v_w, al_p, wall_type)
    wp = 1.0 # Nominal value - will be rescaled later
    wm = wp/enthalpy_ratio(vfm_w, vfp_w) # enthalpy just behind wall

    # Set up parts outside shell where v=0. 2 points only.
    xif = np.linspace(v_wall + dxi,1.0,2)
    vf = np.zeros_like(xif)
    wf = np.ones_like(xif)*wp 

    xib = np.linspace(min(cs2_fun(w_n)**0.5,v_w)-dxi,0.0,2)
    vb = np.zeros_like(xib)
    wb = np.ones_like(xib)*wm 

    # Integrate forward and find shock.
    if not wall_type == 'Detonation':
    # First go
        v,w,xi,t = fluid_integrate_param(vfp_p, wp, v_w+dxi, -TENDDEFAULT, NPDEFAULT, cs2_fun)
        v, w, xi, t = fluid_wall_to_shock(v, w, xi, t, wall_type)
    # Now refine so that there are ~N points between wall and shock
        t_end_refine = t[-1]
        v,w,xi,t = fluid_integrate_param(vfp_p, wp, v_w+dxi, t_end_refine, npts, cs2_fun)
        v, w, xi, t = fluid_wall_to_shock(v, w, xi, t, wall_type)
        v, w, xi = shock_zoom_last_element(v, w, xi)
    # Now complete to xi = 1
        vf = np.concatenate((v,vf))
    # enthalpy
        vfp_s = xi[-1]        # Fluid velocity just ahead of shock in shock frame = shock speed
        vfm_s = 1/(3*vfp_s)   # Fluid velocity just behind shock in shock frame
        wf = np.ones_like(xif)*w[-1]*enthalpy_ratio(vfm_s, vfp_s)
        wf = np.concatenate((w,wf))
    # xi
        xif[0] = xi[-1]
        xif = np.concatenate((xi,xif))
        
    # Integrate backward to sound speed.
    if not wall_type == 'Deflagration':
    # First go
        v,w,xi,t = fluid_integrate_param(vfm_p, wm, v_w-dxi, -TENDDEFAULT, NPDEFAULT, cs2_fun)
        v, w, xi, t = fluid_wall_to_cs(v, w, xi, t, wall_type)
    # Now refine so that there are ~N points between wall and point closest to cs
        t_end_refine = t[-1]
        v,w,xi,t = fluid_integrate_param(vfm_p, wm, v_w-dxi, t_end_refine, npts, cs2_fun)
        v, w, xi, t = fluid_wall_to_cs(v, w, xi, t, wall_type)

    # Now complete to xi = 0
        vb = np.concatenate((v,vb))
        wb = np.ones_like(xib)*w[-1]
        wb = np.concatenate((w,wb))
        xib[0] = cs2_fun(w[-1])**0.5
        xib = np.concatenate((xi,xib))

    # Now put halves together in right order
    v  = np.concatenate((np.flip(vb,0),vf))
    w  = np.concatenate((np.flip(wb,0),wf))
    w  = w*(w_n/w[-1])
    xi = np.concatenate((np.flip(xib,0),xif))

    return v, w, xi


def fluid_wall_to_cs(v, w, xi, t, wall_type, dxi_lim=dxi_small, cs2_fun=cs2_bag):
# Picks out fluid variables which are not too close to fixed point at 
    # xi = cs(w)
    dxi2 = xi**2 - cs2_fun(w)
    if not (wall_type == "Deflagration"):
        relevant = np.where(dxi2 > dxi_lim**2)
        
    return v[relevant], w[relevant], xi[relevant], t[relevant]


def fluid_wall_to_shock(v, w, xi, t, wall_type):
# Truncates fluid variables so last element is just ahead of shock
    n_shock = -2
    if not wall_type == 'Detonation':
        it = np.nditer([v, xi], flags=['c_index'])
        for vv, x in it:
            if vv <= v_shock(x):
                n_shock = it.index
                break
        
    return v[:n_shock+1], w[:n_shock+1], xi[:n_shock+1], t[:n_shock+1]


# Functions for alpha_n (strength parameter at nucleation temp) and 
# alpha_p(lus) (strength parameter just in front of wall)
    
def find_alpha_n(v_wall, alpha_p, wall_type="Calculate", npts=NPDEFAULT):
    # Calculates alpha_N ([(3/4) difference in trace anomaly]/enthalpy) from alpha_plus (a_p)
    # v_wall can be scalar or iterable.
    # alpha_p must be scalar.
    if wall_type == "Calculate":
        wall_type = identify_wall_type_alpha_plus(v_wall, alpha_p)
    _, w, xi = fluid_shell_alpha_plus(v_wall, alpha_p, wall_type, npts)
    n_wall = find_v_index(xi, v_wall)
    return alpha_p*w[n_wall]/w[-1]


def find_alpha_plus(v_wall, alpha_n_given, npts=NPDEFAULT):
    # Calculate alpha_plus from a given alpha_n and v_wall.
    # v_wall can be scalar or iterable.
    # alpha_n_given must be scalar.

    it = np.nditer([None,v_wall],[],[['writeonly','allocate'],['readonly']])
    for ap, vw in it:
        if alpha_n_given < alpha_n_max_detonation(vw):
        # Must be detonation
            wall_type = 'Detonation'
            ap[...] = alpha_n_given
        else:
            if alpha_n_given < alpha_n_max_deflagration(vw):
                if vw <= cs0:
                    wall_type = "Deflagration"
                else:
                    wall_type = "Hybrid"
                def func(x): 
                    return find_alpha_n(vw, x, wall_type, npts) - alpha_n_given
                a_initial_guess = alpha_plus_initial_guess(vw, alpha_n_given)
                al_p = opt.fsolve(func, a_initial_guess, xtol=find_alpha_plus_tol)[0]
                ap[...] = al_p
            else:
                ap[...] = np.nan

    if isinstance(v_wall,np.ndarray):
        return it.operands[0]
    else:    
        return type(v_wall)(it.operands[0])


def alpha_plus_initial_guess(v_wall, alpha_n_given):
    # Calculating initial guess for alpha_plus from alpha_n.
    # Linear approx between alpha_n_min and alpha_n_max
    # Doesn't do obvious checks like Detonation - needs improving?

    if alpha_n_given < 0.05:
        a_guess = alpha_n_given
    else:
        alpha_plus_min = alpha_plus_min_hybrid(v_wall)
        alpha_plus_max = 1. / 3

        alpha_n_min = alpha_n_min_hybrid(v_wall)
        alpha_n_max = alpha_n_max_deflagration(v_wall)

        slope = (alpha_plus_max - alpha_plus_min) / (alpha_n_max - alpha_n_min)

        a_guess = alpha_plus_min + slope * (alpha_n_given - alpha_n_min)

    return a_guess


def find_alpha_n_from_w_xi(w, xi, v_wall, alpha_p):
    # Calculates alpha_N ([(3/4) difference in trace anomaly]/enthalpy) from alpha_plus (ap_)
    # Assuming one has solution arrays w, xi
    n_wall = find_v_index(xi, v_wall)
    return alpha_p*w[n_wall]/w[-1]
    
    
def alpha_n_max_hybrid(v_wall, npts=NPDEFAULT):
    # Calculates maximum alpha_n for given v_wall, for Hybrid
    wall_type = identify_wall_type_alpha_plus(v_wall, 1./3)
    if wall_type == "Deflagration":
        sys.stderr.write('alpha_n_max_hybrid: error: called with v_wall < cs\n')
        sys.stderr.write('     use alpha_n_max_deflagration instead\n')
        sys.exit(6)

    # Might have been returned as "Detonation, which takes precedence over Hybrid
    wall_type = "Hybrid"
    ap = 1./3 - 1e-8
    _, w, xi = fluid_shell_alpha_plus(v_wall, ap, wall_type, npts)
    n_wall = find_v_index(xi, v_wall)

    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalised to 1 at large xi
    return w[n_wall]*(1./3)


def alpha_n_max_deflagration(v_wall, Np=NPDEFAULT):
    # Calculates maximum alpha_n (relative trace anomaly outside bubble) 
    # for given v_wall, for deflagration.
    # Works for hybrids, as they are supersonic deflagrations

    # wall_type_ = identify_wall_type(v_wall_, 1./3)
    it = np.nditer([None,v_wall],[],[['writeonly','allocate'],['readonly']])
    for ww, vw in it:
        if vw > cs0:
            wall_type = "Hybrid"
        else:
            wall_type = "Deflagration"

        ap = 1./3 - 1e-3
        _, w, xi = fluid_shell_alpha_plus(vw, ap, wall_type, Np)
        n_wall = find_v_index(xi, vw)
        ww[...] = w[n_wall+1]*(1./3)

    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalised to 1 at large xi
    # Need n_wall+1, as w is an integral of v, and lags by 1 step
    if isinstance(v_wall,np.ndarray):
        return it.operands[0]
    else:
        return type(v_wall)(it.operands[0])


def alpha_plus_max_detonation(v_wall):
    # Maximum allowed value of alpha_+ for a detonation with wall speed v_wall
    # Comes from inverting v_w > v_Jouguet
    it = np.nditer([None,v_wall],[],[['writeonly','allocate'],['readonly']])
    for bb, vw in it:
        a = 3*(1-vw**2)
        if vw < cs0:
            b = 0.0
        else:
            b = (1-np.sqrt(3)*vw)**2
        bb[...] = b/a

    if isinstance(v_wall,np.ndarray):
        return it.operands[0]
    else:
        return type(v_wall)(it.operands[0])


def alpha_n_max_detonation(v_wall):
    # Maximum allowed value of alpha_n for a detonation with wall speed v_wall
    # Same as alpha_plus_max_detonation, because alpha_n = alpha_+ for detonation.
    return alpha_plus_max_detonation(v_wall)


def alpha_plus_min_hybrid(v_wall):
    # Minimum allowed value of alpha_+ for a hybrid with wall speed v_wall
    # Condition from coincidence of wall and shock
    b = (1-np.sqrt(3)*v_wall)**2
    c = 9*v_wall**2 - 1
    # for bb, vw in np.nditer([b,v_wall_]):
    if isinstance(b, np.ndarray):
        b[np.where(v_wall < 1./np.sqrt(3))] = 0.0
    else:
        if v_wall < cs0:
            b = 0.0
    return b/c


def alpha_n_min_hybrid(v_wall):
    # Minimum alpha_n for a hybrid is equal to maximum alpha_n for a detonation
    return alpha_n_max_detonation(v_wall)


# Functions for calculating approximate solutions
    
def xi_zero(v_wall, v_xi_wall):
    # Used in approximate solution near v(xi) = xi: defined as solution to v(xi0) = xi0
    xi0 = (v_xi_wall + 2*v_wall)/3.
    return xi0


def v_approx_high_alpha(xi, v_wall, v_xi_wall):
    # Approximate solution for v(xi) near v(xi) = xi
    xi0 = xi_zero(v_wall, v_xi_wall)
    A2 = 3*(2*xi0 - 1)/(1 - xi0**2)
    dv = (xi - xi0)
    return xi0 - 2*dv - A2*dv**2


def v_approx_hybrid(xi, v_wall, v_xiWall):
    # Approximate solution for v(xi) near v(xi) = xi (same as v_approx_high_alpha)
    xi0 = (v_xiWall + 2*v_wall)/3.
    A2 = 3*(2*xi0 - 1)/(1 - xi0**2)
    dv = (xi - xi0)
    return xi - 3*dv - A2*dv**2


def w_approx_high_alpha(xi, v_wall, v_xi_wall, w_xi_wall):
    # Approximate solution for w(xi) near v(xi) = xi
    xi0 = xi_zero(v_wall, v_xi_wall)
    return w_xi_wall*np.exp(-12*(xi - xi0)**2/(1 - xi0**2)**2)


def v_approx_low_alpha(xi, v_wall, alpha):
    # Approximate solution for v(xi) at low alpha_+ = alpha_n

    def v_approx_fun(x, v_w):
        # Shape of linearised solution for v(xi)
        return (v_w/x)**2 * (cs0**2 - x**2)/(cs0**2 - v_w**2)

    v_app = np.zeros_like(xi)
    v_max = 3 * alpha * v_wall/abs(3*v_wall**2 - 1)
    shell = np.where(np.logical_and(xi > min(v_wall, cs0), xi < max(v_wall, cs0)))
    v_app[shell] = v_max * v_approx_fun(xi[shell], v_wall)

    return v_app


def w_approx_low_alpha(xi, v_wall, alpha):
    # Approximate solution for w(xi) at low alpha_+ = alpha_n

    v_max = 3 * alpha * v_wall/abs(3*v_wall**2 - 1)
    gaws2 = 1./(1. - 3*v_wall**2)
    w_app = np.exp(8*v_max * gaws2 * v_wall * ((v_wall/cs0) - 1))

    w_app[np.where(xi > max(v_wall, cs0))] = 1.0
    w_app[np.where(xi < min(v_wall, cs0))] = 0.0

    return w_app

# Functions for calculating quantities derived from solutions

def split_integrate(fun_, v, w, xi, v_wall):
    # Split an integration of a function fun_ of arrays v w xi 
    # according to xi inside or outside the wall (expect discontinuity there)
    inside  = np.where(xi < v_wall) 
    outside = np.where(xi > v_wall) 
    int1 = 0.
    int2 = 0.
    if v[inside].size >= 3:
        int1 = part_integrate(fun_, v, w, xi, inside)
    if v[outside].size >= 3:
        int2 = part_integrate(fun_, v, w, xi, outside)
    return int1, int2


def part_integrate(fun_, v, w, xi, where_in):
    # Integrate a function fun_ of arrays v w xi over selection
    # where_in
    xi_in = xi[where_in]
    v_in = v[where_in]
    w_in = w[where_in]
    integrand = fun_(v_in, w_in, xi_in)
    return np.trapz(integrand, xi_in)


def de_from_w(w, xi, v_wall, alpha_n):
#   Calculates energy density from enthalpy, assuming
#   bag equation of state, and returns difference from undisturbed value at large r.
#    alpha_n = find_alpha_n_from_w_xi(w,xi,v_wall,alpha_p) 
    e_from_w = e(w, phase(xi,v_wall), 0.75*w*alpha_n)
              
    return e_from_w - e_from_w[-1]


def mean_energy_change(v, w, xi, v_wall, alpha_n):
    # Compute change in energy density in bubble relative to outside value
#    def ene_diff(v,w,xi):
#        return de_from_w(w, xi, v_wall, alpha_n)
#    int1, int2 = split_integrate(ene_diff, v, w, xi**3, v_wall)
#    integral = int1 + int2
    integral = np.trapz(de_from_w(w, xi, v_wall, alpha_n), xi**3)
    return integral / v_wall**3


def mean_enthalpy_change(v, w, xi, v_wall):
    # Compute mean change in enthalphy in bubble relative to outside value
#    def en_diff(v, dw, xi):
#        return dw
#    int1, int2 = split_integrate(en_diff, v, w - w[-1], xi**3, v_wall)
#    integral = int1 + int2
    integral = np.trapz((w - w[-1]), xi**3)
    return integral / v_wall**3


def Ubarf_squared(v, w, xi, v_wall):
    # Compute mean square (4-)velocity of fluid in bubble
#    def fun(v,w,xi):
#        return w * v**2 * gamma2(v)
#    int1, int2 = split_integrate(fun, v, w, xi**3, v_wall)
#    integral = int1 + int2
    integral = np.trapz(w * v**2 * gamma2(v), xi**3)
    
    return integral / (w[-1]*v_wall**3)


def get_ke_frac(v_wall, alpha_n, npts=NPDEFAULT):
    # Determine kinetic energy fraction (of total energy)
    # Bag equation of state only so far.
    # Note e_n = (3./4) w_n (1+alpha_n) (assuming no trace anomaly in broken phase)
    ubar2 = get_ubarf2(v_wall, alpha_n, npts)

    return ubar2/(0.75*(1 + alpha_n))


def get_ubarf2(v_wall, alpha_n, npts=NPDEFAULT):
    # Get mean square fluid velocity. 
    # v_wall can be scalar or iterable.
    # alpha_n must be scalar
    it = np.nditer([v_wall, None])
    for vw, Ubarf2 in it:
        wall_type = identify_wall_type(vw, alpha_n)
        if not wall_type=='Error':
            # Now ready to solve for fluid profile
            v, w, xi = fluid_shell(vw, alpha_n, npts)
            Ubarf2[...] = Ubarf_squared(v, w, xi, vw)
        else:
            Ubarf2[...] = np.nan

    # Ubarf2 is stored in it.operands[1]
    return it.operands[1]


def get_kappa(v_wall, alpha_n, npts=NPDEFAULT, verbosity=0):
    # Calculates efficiency factor kappa from wall velocity (can be array)
    # NB was called get_kappa_arr
    it = np.nditer([v_wall, None])
    for vw, kappa in it:
        wall_type = identify_wall_type(vw, alpha_n)

        if not wall_type=='Error':
            # Now ready to solve for fluid profile
            v, w, xi = fluid_shell(vw, alpha_n, npts)

            kappa[...] = Ubarf_squared(v, w, xi, vw)/(0.75*alpha_n)
        else:
            kappa[...] = np.nan
        if verbosity > 0:
            sys.stderr.write("{:8.6f} {:8.6f} {} ".format(vw, alpha_n, kappa),flush=True)

    if isinstance(v_wall,np.ndarray):
        kappa_out = it.operands[1]
    else:
        kappa_out = type(v_wall)(it.operands[1])

    return kappa_out


def get_kappa_de(v_wall, alpha_n, npts=NPDEFAULT, verbosity=0):
    # Calculates efficiency factor kappa and fractional change in energy 
    # from wall velocity array. Sum should be 1 (bag model)
    it = np.nditer([v_wall, None, None])
    for vw, kappa, de in it:
        wall_type = identify_wall_type(vw, alpha_n)

        if not wall_type=='Error':
            # Now ready to solve for fluid profile
            v, w, xi = fluid_shell(vw, alpha_n, npts)
            # Esp+ epsilon is alpha_n * 0.75*w_n
            kappa[...] = Ubarf_squared(v, w, xi, vw)/(0.75 * alpha_n)
            de[...]    = mean_energy_change(v, w, xi, vw, alpha_n)/(0.75 * alpha_n * w[-1])
        else:
            kappa[...] = np.nan
            de[...] = np.nan
        if verbosity > 0:
            sys.stderr.write("{:8.6f} {:8.6f} {} {}".format(vw, alpha_n, kappa, de),flush=True)

    if isinstance(v_wall,np.ndarray):
        kappa_out = it.operands[1]
        de_out = it.operands[2]
    else:
        kappa_out = type(v_wall)(it.operands[1])
        de_out = type(v_wall)(it.operands[2])

    return kappa_out, de_out


def get_kappa_dw(v_wall, alpha_n, npts=NPDEFAULT, verbosity=0):
    # Calculates efficiency factor kappa and fractional change in 0.75*emthalpy 
    # from wall velocity array. Sum should be 1.
    it = np.nditer([v_wall, None, None])
    for vw, kappa, dw in it:
        wall_type = identify_wall_type(vw, alpha_n)

        if not wall_type=='Error':
            # Now ready to solve for fluid profile
            v, w, xi = fluid_shell(vw, alpha_n, npts)
            # Esp+ epsilon is alpha_n * 0.75*w_n
            kappa[...] = Ubarf_squared(v, w, xi, vw)/(0.75 * alpha_n)
            dw[...]    = 0.75 * mean_enthalpy_change(v, w, xi, vw)/(0.75 * alpha_n * w[-1])
        else:
            kappa[...] = np.nan
            dw[...] = np.nan
        if verbosity > 0:
            sys.stderr.write("{:8.6f} {:8.6f} {} {}".format(vw, alpha_n, kappa, dw),flush=True)

    if isinstance(v_wall,np.ndarray):
        kappa_out = it.operands[1]
        dw_out = it.operands[2]
    else:
        kappa_out = type(v_wall)(it.operands[1])
        dw_out = type(v_wall)(it.operands[2])

    return kappa_out, dw_out

