# Toolbox file for EIKR EoS, lifted from existing Toolbox as needed
import numpy as np


def phi_broken(T):
    return(params[2]*T + np.sqrt((params[2]**2)*(T**2)-4*params[1]*((T**2)-(params[4]**2))*params[3]))/(2*params[3])


def dp_dT(T):
    return (4./3.)*a0*T**3 - params[1]*(T**2)*(phi_broken(T)**2) - (1./3.)*params[2]*phi_broken(T)**3


def de_dT(T):
    return 4.*a0*(T**3) - params[1]*T*(phi_broken(T)**3)


def cs(T):
    return np.sqrt((dp_dT(T))/(de_dT(T)))

gstar = 106.75
D = 0.4444
A = 0.1990
lamda = 0.0396
T0_Tc = 1./(np.sqrt(2))
Tn_Tc = 0.8

params = (gstar, D, A, lamda, T0_Tc, Tn_Tc)
a0 = (params[0]*(np.pi**2)/30)


