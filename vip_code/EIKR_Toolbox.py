# Toolbox file for EIKR EoS, lifted from existing Toolbox as needed
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Mechanics_Toolbox as Mech


def phi_broken(T):
    # print 'phi:'
    # print 'T = ', T
    # print ''
    # if (A*T)**2-4*lamda*D*(T**2-T0_Tc**2)>0:
    #     pass
    # else:
    #     print 'Error: T**2-T0**2 = ', (T**2-T0_Tc**2)
    #     print 'T = ', T
    #     print 'T0 = ', T0_Tc
    #     sys.exit(1)
    return(A*T+np.sqrt((A*T)**2-4*lamda*D*(T**2-T0_Tc**2)))/(2*lamda)


def v(T):
    return V0 + (1./2.)*D*(T**2-T0_Tc**2)*phi_broken(T)**2 - (1./3.)*A*T*phi_broken(T)**3 + \
           (1./4.)*lamda*phi_broken(T)**4


def dv_dt(T):
    # print 'dvdt:'
    # print 'T = ', T
    # print ''
    return D*T*phi_broken(T)**2 - (1./3.)*A*phi_broken(T)**3


def d2v_dt2(T):
    return D*phi_broken(T)**2


# All thermodynamic quantities in broken phase (minus) unless stated otherwise
def p(T):
    # print 'p:'
    # print 'T = ', T
    # print''
    return 1./3.*a0*T**4 - v(T)


def s(T):
    return (4./3.)*a0*T**3 - dv_dt(T)


def w_minus(T):
    return T*s(T)


def w_plus(T):
    return (4. / 3.) * a0 * T ** 4


def vminus(T,Xiw):
    B = ((Xiw**2)*(1+alphaplus(T))**2-alphaplus(T)**2 - (2./3.)*alphaplus(T)+(1./3.))/(2*Xiw*(1+alphaplus(T)))
#    A = Xiw
#    B = - ( Xiw**2 + (1./3) - (1 - Xiw**2)*alphaplus(T) )
#    C = Xiw/3
#    return  (-B + np.sqrt(B**2 - 4*A*C) )/(2.*A)
    return B + np.sqrt(B**2 - 1./3)


def energy_cons(T, Xiw):
    # Expression from EM conservation equation to be equated to broken phase enthalpy
    return (Xiw*(1/(1-Xiw**2))*w_plus(Tn_Tc))/(vminus(T, Xiw)*(1/(1-vminus(T, Xiw)**2)))


def delta_w(T, Xiw):
    # print'delta_w:'
    # print 'T = ', T
    # print 'Xiw =', Xiw
    dw = w_minus(T) - energy_cons(T, Xiw)
    # print 'Returning ', dw
    # print ''
    # Function whose root gives temperature on broken phase side of wall
    return dw


def e(T):
    return w_minus(T) - p(T)


def de_dT(T):
    return T*(4.*a0*T**2 - d2v_dt2(T))


def cs2(T):
    return s(T)/de_dT(T)


def cs(T):
    return np.sqrt(cs2(T))


def alphaplus(T):
    return epsilon(T)/(a0*Tn_Tc**4)  # Caution! Detonation only


def epsilon(T):
    return 0.25*(e(T)-3*p(T))


def tps_from_wps(tms, vms, vps):
    g_m_shock2 = Mech.gamma(vms)**2
    g_p_shock2 = Mech.gamma(vps)**2
    wps = (w_minus(tms)*g_m_shock2*vms)/(g_p_shock2*vps)
    return ((3./4.)*(wps/a0))**0.25


gstar = 106.75
D = 0.4444
A = 0.1990
lamda = 0.0396
T0_Tc = 1./(np.sqrt(2))
Tn_Tc = 0.8

V0 = (D*T0_Tc**2)**2/(4*lamda)
a0 = (gstar*(np.pi**2)/30)

# Ts = np.linspace(0, 1, 100)
# ys = e(Ts) - 3*p(Ts)
#
# plt.figure()
# plt.plot(Ts, ys)
# plt.show()