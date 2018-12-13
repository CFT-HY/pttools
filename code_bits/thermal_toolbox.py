import numpy as np

#weak
#gstar = 106.75
#gamma = 0.4444
#A = 0.1990
#lamda = 0.0396
#T0_Tc = 1./(np.sqrt(2))
#Tn_Tc = 0.86

#intermediate
#gamma = 0.2222
#A = 0.1990
#lamda = 0.0792
#T0_Tc = 1./(np.sqrt(2))
#Tn_Tc = 0.8

# strong
gstar=106.75
gamma=0.66667
A = 0.1990232604
lamda=0.0264068
T0_Tc=0.70710678118654757
Tn_Tc = 0.77278

params = (gstar, gamma, A, lamda, T0_Tc, Tn_Tc)

a0 = (gstar*(np.pi**2)/30)

phi0_Tc = T0_Tc*np.sqrt(gamma/lamda)
V0_Tc4 = 0.25*lamda*phi0_Tc**4

def phi_broken(T):
	return ((A*T + np.sqrt((A**2)*(T**2) - 4*gamma*((T**2)-(T0_Tc**2))*lamda))/(2*lamda))

def V(T):
	return V0_Tc4 + (1./2.)*gamma*(T**2 - T0_Tc**2)*phi_broken(T)**2 - (1./3.)*A*T*phi_broken(T)**3 + (1./4.)*lamda*phi_broken(T)**4

def dV_dT(T):
	return gamma*T*phi_broken(T)**2 - (1./3.)*A*phi_broken(T)**3

def p(T):
	return (1./3.)*a0*(T**4) - V(T)

def dp_dT(T):
	return (4./3.)*a0*T**3 - dV_dT(T)

def e(T):
	return a0*(T**4) + V(T) - T*dV_dT(T)

def de_dT(T):
	return 4.*a0*(T**3) - gamma*T*(phi_broken(T)**2)

def w_broken(T):
	return e(T) + p(T)

def cs(T):
	return np.sqrt((dp_dT(T))/(de_dT(T)))

def cs2(T):
	return dp_dT(T)/de_dT(T)

def epsilon(T):
	return 0.25*(e(T)-3*p(T))

def wplus(T):
	return (4./3.)*a0*T**4

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

def wminus(T,Xiw):
	return (Xiw*(1/(1-Xiw**2))*wplus(Tn_Tc))/(vminus(T,Xiw)*(1/(1-vminus(T,Xiw)**2)))

def wminus_diff(T,Xiw):
	return w_broken(T) - wminus(T,Xiw)

def vJouguet(T):
    ap = alphaplus(T)
    return cs(T)*(np.sqrt(ap*(2+3*ap)) + 1)/(1+ap)

