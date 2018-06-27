# Toolbox file for EIKR EoS, lifted from existing Toolbox as needed
from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import Mechanics_Toolbox as Mech


def set_params(name):
    global gstar, D, A, lamda, T0, Tn, m0_2, mu0
    if name == 'weak':
        gstar = 106.75
        D = 0.4444
        A = 0.1990
        lamda = 0.0396
        T0 = 1./(np.sqrt(2))
        Tn = 0.86
        m0_2 = -D*T0**2
        mu0 = 0
    elif name == 'intermediate':
        D = 0.2222
        A = 0.1990
        lamda = 0.0792
        T0 = 1./(np.sqrt(2))
        Tn = 0.8
        m0_2 = -D*T0**2
        mu0 = 0
    elif name == 'strong':
        gstar=106.75
        D=0.66667
        A = 0.1990232604
        lamda=0.0264068
        T0=1./(np.sqrt(2))
        Tn = 0.77278
        m0_2 = -D*T0**2
        mu0 = 0
    else:
        sys.exit('set_params: params name not recognised')
        
    compute_derived_params()

    print("set_params: name  ", name)
    print_params()
    
    return 1


def print_params():
    print("print_params:")
    print("gstar ", gstar)
    print("a0    ", a0)
    print("V00   ", V00)
    print("D     ", D)
    print("m0_2  ", m0_2)
    print("A     ", A)
    print("mu0   ", mu0)
    print("lamda ", lamda)
    print("T0    ", T0)
    print("Tn    ", Tn)
    
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


def dV_dT(T):
    # Derivative of effective potential wrt temperature
    return D*T*phi_broken(T)**2 - (1./3.)*A*phi_broken(T)**3


def d2V_dT2(T):
    # Second derivative of effective potential wrt temperature
    return D*phi_broken(T)**2


def dV_dphi(T,phi=None):
    if phi is None:
        phi=phi_broken(T)
    # Effective potential
    return (D*T**2+m0_2)*phi + \
            (-A*T+mu0)*phi**2 + \
            lamda*phi**3


def d2V_dphi2(T,phi=None):
    if phi is None:
        phi=phi_broken(T)
    # Effective potential
    return (D*T**2+m0_2) + \
            2*(-A*T+mu0)*phi + \
            3*lamda*phi**2


# All thermodynamic quantities in broken phase (minus) unless stated otherwise
def T_w(w):
    return ((3.*w)/(4.*a0))**0.25


def p(T):
    # Equilibrium pressure
    return 1./3.*a0*T**4 - V(T)


def p_w(w, e):
    T = T_w(w)
    return 1. / 3. * a0 * T ** 4 - V(T)


def s(T):
    # Equilibrium entropy
    return (4./3.)*a0*T**3 - dV_dT(T)


def s_w(w):
    T = T_w(w)
    return (4./3.)*a0*T**3 - dV_dT(T)


def w_minus(T):
    # Equilibrium enthalpy
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


def e(T):
    # Equilibrium energy density
    return w_minus(T) - p(T)


def de_dT(T):
    # Equilibrium specific heat
    return T*(4.*a0*T**2 - d2V_dT2(T))


def de_dT_w(w):
    T = T_w(w)
    return T*(4.*a0*T**2 - d2V_dT2(T))


def cs2(T):
    # Equilibrium sound speed squared
    return s(T)/de_dT(T)


def cs(T):
    # Equilibrium sound speed
    return np.sqrt(cs2(T))


def alphaplus(T):
    # Equilibrium transition strength parameter
    return epsilon(T)/(a0*Tn**4)  # Caution! Detonation only


def epsilon(T):
    # 0.25 * equilibrium trace anomaly
    return 0.25*(e(T)-3*p(T))


def tps_from_wps(tms, vms, vps):
    g_m_shock2 = Mech.gamma(vms)**2
    g_p_shock2 = Mech.gamma(vps)**2
    wps = (w_minus(tms)*g_m_shock2*vms)/(g_p_shock2*vps)
    return ((3./4.)*(wps/a0))**0.25


set_params('weak')
