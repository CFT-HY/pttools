# Toolbox file for EIKR EoS, lifted from existing Toolbox as needed
from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import scipy.optimize as opt
import Mechanics_Toolbox as Mech

def set_params(name, new_value=None):
    global gstar, D, A, lamda, T0, Tn, m0_2, mu0
    if name == 'weak':
        gstar = 106.75
        D = 2/9.
        A = 0.1990232604
        lamda = 0.0792
        T0 = 1./(np.sqrt(2))
        Tn = 0.86
        m0_2 = -D*T0**2
        mu0 = 0
    elif name == 'intermediate':
        gstar = 106.75
        D = 4/9.
        A = 0.1990232604
        lamda = 0.0396
        T0 = 1./(np.sqrt(2))
        Tn = 0.8
        m0_2 = -D*T0**2
        mu0 = 0
    elif name == 'strong':
        gstar=106.75
        D=2/3.
        A = 0.1990232604
        lamda=0.0264068
        T0=1./(np.sqrt(2))
        Tn = 0.77278
        m0_2 = -D*T0**2
        mu0 = 0
    elif name == 'vstrong':
#        gamma=0.2222*4
#        T_0=0.70710678118654757
#        Lambda=0.079221/4
#        gstar=106.75
#        alpha=0.19902
#        T_N=0.7577
        gstar=106.75
        D=8/9.
        A = 0.1990232604
        lamda=0.079221/4
        T0=1./(np.sqrt(2))
        Tn = 0.7577
        m0_2 = -D*T0**2
        mu0 = 0
    elif name == 'gstar':
        gstar = new_value
    elif name == 'D':
        D = new_value
    elif name == 'A':
        A = new_value
    elif name == 'lamda':
        lamda = new_value
    elif name == 'T0':
        T0 = new_value
    elif name == 'Tn':
        Tn = new_value
    elif name == 'm0_2':
        m0_2 = new_value
    elif name == 'mu0':
        mu0 = new_value
            
    else:
        sys.exit('set_params_eikr: params name not recognised')
        
    compute_derived_params()

    # print("set_params: name  ", name)
    # print_params()
    
    return 1


def print_params():
    print("print_params: basic")
    print("gstar ", gstar)
    print("D     ", D)
    print("m0_2  ", m0_2)
    print("A     ", A)
    print("mu0   ", mu0)
    print("lamda ", lamda)
    print("T0    ", T0)
    print("Tn    ", Tn)
    print("print_params: derived")
    print("a0    ", a0)
    print("V00   ", V00)
    print("Tcrit ", Tcrit())
    return 1
    

def compute_derived_params():
    global V00, a0
    V00 = -V0()
    a0 = (gstar*(np.pi**2)/30)
#    print("compute_derived_params: V00  ", V00)
#    print("compute_derived_params: a0   ", a0)
    return 1


def phi_broken(T):
    # Broken phase equilibrium scalar value
    return (A*T - mu0 + np.sqrt((A*T - mu0)**2-4*lamda*(D*T**2+m0_2)))/(2*lamda)


def Tcrit():
    a = lamda*D - (2/9.)*A**2
    b = +(4/9.)*mu0*A
    c = lamda*m0_2 - (2/9.)*mu0**2
    return (0.5/a)*(-b + (b**2 - 4*a*c)**0.5)

    
def V0():
    # Effective potential at zero temperature
    return 0.5*(m0_2)*phi_broken(0)**2 + \
            (1./3.)*(mu0)*phi_broken(0)**3 + \
            0.25*lamda*phi_broken(0)**4


def V(T,phi=None):
    if phi is None:
        phi=phi_broken(T)
    # Effective potential
    return V00 + 0.5*(D*T**2+m0_2)*phi**2 + \
            (1./3.)*(-A*T+mu0)*phi**3 + \
            0.25*lamda*phi**4


def dV_dT(T,phi=None):
    # Derivative of effective potential wrt temperature
    if phi is None:
        phi=phi_broken(T)
    return D*T*phi**2 - (1./3.)*A*phi**3


def d2V_dT2(T,phi=None):
    # Second derivative of effective potential wrt temperature
    if phi is None:
        phi=phi_broken(T)
    return D*phi**2


def dV_dphi(T,phi=None):
    if phi is None:
        phi=phi_broken(T)
    return (D*T**2+m0_2)*phi + \
            (-A*T+mu0)*phi**2 + \
            lamda*phi**3


def d2V_dphi2(T,phi=None):
    if phi is None:
        phi=phi_broken(T)
    return (D*T**2+m0_2) + \
            2*(-A*T+mu0)*phi + \
            3*lamda*phi**2

def d2V_dTdphi(T,phi=None):
    if phi is None:
        phi=phi_broken(T)
    return 2*D*T*phi - A*T*phi**2

    
def dphi_dT(T,phi=None):
    if phi is None:
        phi=phi_broken(T)
        return -(2*D*T - A*phi)/(2*lamda*phi + mu0 - A*T)
    else:
        return np.zeros_like(T)

        
# All thermodynamic quantities in broken phase (minus) unless stated otherwise
def T_w(w, phi=None):
    Tguess0 = ((3.*w)/(4.*a0))**0.25
    Tguess1 = (3.*(w + Tguess0*dV_dT(Tguess0,phi))/(4.*a0))**0.25
#    print(T0,T1)
    def Twfun(T):
        return w - (4./3)*a0*T**4 + T*dV_dT(T,phi)
    # Think about Newton-Raphson for the solution?
    sol = opt.fsolve(Twfun, Tguess1)
    if isinstance(w,np.ndarray):
        return sol
    else:
        return sol[0]
        

def p(T, phi=None):
    # Equilibrium pressure
    return 1./3.*a0*T**4 - V(T,phi)


def p_w(w, phi=None):
    T = T_w(w, phi)
    return p(T, phi)


def s(T, phi=None):
    # Equilibrium entropy
    return (4./3.)*a0*T**3 - dV_dT(T, phi)


def s_w(w, phi=None):
    T = T_w(w, phi)
    return s(T, phi)

    
def w(T, phi=None):
    # Equilibrium enthalpy
    return T*s(T, phi)

    
def w_minus(T):
    # Equilibrium enthalpy, broken phase
    return T*s(T)


def w_plus(T):
    # Equilibrium enthalpy, symmetric phase
    return (4. / 3.) * a0 * T ** 4


def vminus(T,xi_w):
    # Fluid speed just behind wall, detonation
    B = ((xi_w**2)*(1+alphaplus(T))**2-alphaplus(T)**2 - (2./3.)*alphaplus(T)+(1./3.))/(2*xi_w*(1+alphaplus(T)))
#    A = xi_w
#    B = - ( xi_w**2 + (1./3) - (1 - xi_w**2)*alphaplus(T) )
#    C = xi_w/3
#    return  (-B + np.sqrt(B**2 - 4*A*C) )/(2.*A)
    return B + np.sqrt(B**2 - 1./3)


def energy_cons(T, xi_w):
    # Expression from EM conservation equation to be equated to broken phase enthalpy
    return (xi_w*(1/(1-xi_w**2))*w_plus(Tn))/(vminus(T, xi_w)*(1/(1-vminus(T, xi_w)**2)))


def delta_w(T, xi_w):
    # print'delta_w:'
    # print 'T = ', T
    # print 'xi_w =', xi_w
    dw = w_minus(T) - energy_cons(T, xi_w)
    # print 'Returning ', dw
    # print ''
    # Function whose root gives temperature on broken phase side of wall
    return dw


def e(T, phi=None):
    # Equilibrium energy density
    return w(T, phi) - p(T, phi)


def de_dT(T, phi=None):
    # Equilibrium specific heat
    return T*(4.*a0*T**2 - d2V_dT2(T, phi) 
            - d2V_dTdphi(T,phi)*dphi_dT(T,phi))


def de_dT_w(w, phi=None):
    # Equilibrium specific heat (fn of w)
    T = T_w(w, phi)
    return de_dT(T, phi)


def cs2(T, phi=None):
    # Equilibrium sound speed squared
    return s(T, phi)/de_dT(T, phi)


def cs2_w(w, phi=None):
    # Equilibrium sound speed squared (fn of w)
    return s_w(w, phi)/de_dT_w(w, phi)


def cs(T, phi=None):
    # Equilibrium sound speed
    return np.sqrt(cs2(T, phi))


def cs_w(w, phi=None):
    # Equilibrium sound speed (fn of w)
    return np.sqrt(cs2_w(w, phi))


def epsilon(T, phi=None):
    # 0.25 * equilibrium trace anomaly
    return 0.25*(e(T, phi)-3*p(T, phi))


def epsilon_w(w, phi=None):
    # 0.25 * equilibrium trace anomaly (fn of w)
    T = T_w(w, phi)
    return 0.25 * (e(T, phi) - 3 * p(T, phi))

    
def alphaplus(T):
    # Equilibrium transition strength parameter
    return (epsilon(T,0)-epsilon(T))/(0.75*w(T,0))


def a(T, phi=None):
    return 0.75*w(T, phi)/T**4


def tps_from_wps(tms, vms, vps):
    g_m_shock2 = Mech.gamma(vms)**2
    g_p_shock2 = Mech.gamma(vps)**2
    wps = (w_minus(tms)*g_m_shock2*vms)/(g_p_shock2*vps)
    return ((3./4.)*(wps/a0))**0.25


set_params('weak')
