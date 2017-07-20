# Toolbox file for EIKR EoS, lifted from existing Toolbox as needed
import numpy as np


def phi_broken(T):
    return(params[2]*T + np.sqrt((params[2]**2)*(T**2)-4*params[1]*((T**2)-(params[4]**2))*params[3]))/(2*params[3])


def dp_dT(T):
    return (4./3.)*a0*T**3 - params[1]*(T**2)*(phi_broken(T)**2) - (1./3.)*params[2]*phi_broken(T)**3


def de_dT(T):
    return 4.*a0*(T**3) - params[1]*T*(phi_broken(T)**2)


def cs(T):
    return np.sqrt(cs2(T))


def cs2(T):
    return (dp_dT(T))/(de_dT(T))


def alphaplus(T):
    return epsilon(T)/(a0*Tn_Tc**4)  # Caution! Detonation only


def epsilon(T):
    return 0.25*(e(T)-3*p(T))


def p(T):
    return (1./3.)*a0*(T**4)-(1./2.)*params[1]*(T**2-params[4]**2)*phi_broken(T)**2 + (1./3.)*params[2]*T*phi_broken(T)\
        **3 - (1./4.)*params[3]*phi_broken(T)**4 - V0


def e(T):
    return a0*(T**4) - (1./2.)*params[1]*(T**2 + params[4]**2)*phi_broken(T)**2 + (1./4.)*params[3]*phi_broken(T)**4 + V0


def s(T):
    return dp_dT(T)


def wminus(T):
# Broken phase enthalpy
    return (4./3.)*a0*(T**4)-params[1]*(T**2)*(phi_broken(T)**2)-(1./3.)*params[2]*T*(phi_broken(T)**3)


def energy_cons(T, Xiw):
# Expression from EM conservation eequation to be equated to broken phase enthalpy
    return (Xiw*(1/(1-Xiw**2))*wplus(Tn_Tc))/(v_minus(T, Xiw)*(1/(1-v_minus(T, Xiw)**2)))


def v_minus(T, Xiw):
    B = ((Xiw**2)*(1+alphaplus(T))**2-alphaplus(T)**2 - (2./3.)*alphaplus(T)+(1./3.))/(2*Xiw*(1+alphaplus(T)))
    return B + np.sqrt(B**2 - 1./3)


def delta_w(T, Xiw):
    # Function whose root gives temperture on borken phase side of wall
    return wminus(T) - energy_cons(T, Xiw)


def wplus(T):
    return (4./3.)*a0*T**4


def mu(vw, v):
    return (vw-v)/(1.-vw*v)


def gamma(v):
    return np.sqrt(1./(1-v**2))


def dy_dxi(y, xi):
    vmp, tm = y
    dv_dxi = (2 * vmp / xi) * 1 / ((1 - vmp * xi) * ((mu(xi, vmp) ** 2 / cs(tm)) - 1))
    #    dlogT_dxi = (2*v / (1-(ga**2)*v*(xi - v))) * 1/((xi - v)/(cs2(logT)*(1-(ga**2)*v*(xi-v))-(mu(xi, v)/(ga**2))))
    dlog_tm_dxi = (gamma(vmp))**2 * mu(xi, vmp) * dv_dxi
    return [dv_dxi, dlog_tm_dxi]



gstar = 106.75
D = 0.4444
A = 0.1990
lamda = 0.0396
T0_Tc = 1./(np.sqrt(2))
Tn_Tc = 0.8

V0 = (D*T0_Tc**2)**2/(4*lamda)

params = (gstar, D, A, lamda, T0_Tc, Tn_Tc)
a0 = (params[0]*(np.pi**2)/30)


