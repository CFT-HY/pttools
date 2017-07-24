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


def p(T):
    return 1./3.*a0*T**4 - v()


gstar = 106.75
D = 0.4444
A = 0.1990
lamda = 0.0396
T0_Tc = 1./(np.sqrt(2))
Tn_Tc = 0.8

V0 = (D*T0_Tc**2)**2/(4*lamda)
a0 = (gstar*(np.pi**2)/30)