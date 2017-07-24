import numpy as np


def phi_broken(T):
    return(A*T+np.sqrt((A*T)**2-4*lamda*D*(T**2-T0_Tc**2)))/2*lamda


def v(T):
    return V0 + (1./2.)*D*(T**2-T0_Tc**2)*phi_broken(T)**2 - (1./3.)*A*T*phi_broken(T)**3 + \
           (1./4.)*lamda*phi_broken(T)**4


def dv_dt(T):
    return D*T*phi_broken(T)**2 - (1./3.)*A*phi_broken(T)**3


def d2v_dt2(T):
    return D*phi_broken(T)**2

# All thermodynamic quantities in broken phase unless stated otherwise
def p(T):
    return 1./3.*a0*T**4 - v(T)


def s(T):
    return (4./3.)*a0*T**3 - dv_dt(T)


def w_minus(T):
    return T*s(T)


def w_plus(T):
    return (4. / 3.) * a0 * T ** 4


def energy_cons(T, Xiw):
# Expression from EM conservation eequation to be equated to broken phase enthalpy
    return (Xiw*(1/(1-Xiw**2))*w_plus(Tn_Tc))/(v_minus(T, Xiw)*(1/(1-v_minus(T, Xiw)**2)))


def delta_w(T, Xiw):
    # Function whose root gives temperature on broken phase side of wall
    return w_minus(T) - energy_cons(T, Xiw)


def e(T):
    return w_minus(T) - p(T)


def de_dt(T):
    return T*(4.*a0*T**2 - d2v_dt2(T))


def cs2(T):
    return s(T)/de_dt(T)


def cs(T):
    return np.sqrt(cs2(T))


def alphaplus(T):
    return epsilon(T)/(a0*Tn_Tc**4)  # Caution! Detonation only


def epsilon(T):
    return 0.25*(e(T)-3*p(T))



def tps_from_wps(tms, vms, vps):
    g_m_shock2 = gamma(vms)**2
    g_p_shock2 = gamma(vps)**2
    wps = (w_minus(tms)*g_m_shock2*vms)/(g_p_shock2*vps)
    return ((3./4.)*(wps/aplus))**0.25


gstar = 106.75
D = 0.4444
A = 0.1990
lamda = 0.0396
T0_Tc = 1./(np.sqrt(2))
Tn_Tc = 0.8

V0 = (D*T0_Tc**2)**2/(4*lamda)
a0 = (gstar*(np.pi**2)/30)