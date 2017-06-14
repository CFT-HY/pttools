import numpy as np
from scipy.optimize import fsolve
gstar = 106.75
#gamma = 0.2222
gamma = 0.4444
A = 0.1990
#lamda = 0.0792
lamda = 0.0396
T0_Tc = 1./(np.sqrt(2))
#Tn_Tc = 0.86
Tn_Tc = 0.8

params = (gstar, gamma, A, lamda, T0_Tc, Tn_Tc)

a0 = (params[0]*(np.pi**2)/30)

def phi_broken(T):
	return ((params[2]*T + np.sqrt((params[2]**2)*(T**2) - 4*params[1]*((T**2)-(params[4]**2))*params[3]))/(2*params[3]))

def wplus(T):
	return (4./3.)*a0*T**4

def wminus1(T):
	return (4./3.)*a0*(T**4)-params[1]*(T**2)*(phi_broken(T)**2)-(1./3.)*params[2]*T*(phi_broken(T)**3)

def V(T):
	return (1./2.)*params[1]*(T**2 - params[4]**2)*phi_broken(T)**2 - (1./3.)*params[2]*T*phi_broken(T)**3 + (1./4.)*params[3]*phi_broken(T)**4

def p(T):
	return (1./3.)*a0*(T**4)-(1./2.)*params[1]*(T**2-params[4]**2)*phi_broken(T)**2 - (1./3.)*params[2]*T*phi_broken(T)**3 + (1./4.)*params[3]*phi_broken(T)**4

def dp_dT(T):
	return (4./3.)*a0*T**3 - params[1]*(T**2)*(phi_broken(T)**2) - (1./3.)*params[2]*phi_broken(T)**3

def e(T):
	return a0*(T**4)-(1./2.)*params[1]*(T**2 - params[4]**2)*phi_broken(T)**2-(1./4.)*params[3]*phi_broken(T)**4

def de_dT(T):
	return 4.*a0*(T**3) - params[1]*T*(phi_broken(T)**3)

def epsilon(T):
	return 0.25*(e(T)-3*p(T))

def alphaplus(T):
    return epsilon(T)/(a0*Tn_Tc**4) # Caution! Detonation only

#def vminus(T,Xiw):
#    return ((Xiw**2)*(1+alphaplus(T)2)**2-alphaplus(T)**2+(2./3.)*alphaplus(T)+(1./3.))/(2*Xiw*(1+alphaplus(T))) + np.sqrt((((Xiw**2)*(1+alphaplus(T)**2)-alphaplus(T)**2-(2./3.)*alphaplus(T)+(1./3.))/(2*Xiw*(1+alphaplus(T))))**2 - (1./3.))

def vminus(T,Xiw):
    B = ((Xiw**2)*(1+alphaplus(T))**2-alphaplus(T)**2 - (2./3.)*alphaplus(T)+(1./3.))/(2*Xiw*(1+alphaplus(T)))
#    A = Xiw
#    B = - ( Xiw**2 + (1./3) - (1 - Xiw**2)*alphaplus(T) )
#    C = Xiw/3
#    return  (-B + np.sqrt(B**2 - 4*A*C) )/(2.*A)
    return  (B + np.sqrt(B**2 - 1./3) )

def wminus2(T,Xiw):
	return (Xiw*(1/(1-Xiw**2))*wplus(Tn_Tc))/(vminus(T,Xiw)*(1/(1-vminus(T,Xiw)**2)))

def wminus3(T,Xiw):
	return wminus1(T) - wminus2(T,Xiw)

def cs(T):
	return np.sqrt((dp_dT(T))/(de_dT(T)))

def vJouguet(T):
    ap = alphaplus(T)
    return cs(T)*(np.sqrt(ap*(2+3*ap)) + 1)/(1+ap)

