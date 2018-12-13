import numpy as np


def epsilon(T):
    return 0.25*(e(T)-3*p(T))


def e(T):
    return a0*(T**4)-(1./2.)*params[1]*(T**2 - params[4]**2)*phi_broken(T)**2-(1./4.)*params[3]*phi_broken(T)**4


def p(T):
    return (1./3.)*a0*(T**4)-(1./2.)*params[1]*(T**2-params[4]**2)*phi_broken(T)**2 - (1./3.)*params[2]*T*phi_broken(T)\
        **3 + (1./4.)*params[3]*phi_broken(T)**4


def phi_broken(T):
    return(params[2]*T + np.sqrt((params[2]**2)*(T**2)-4*params[1]*((T**2)-(params[4]**2))*params[3]))/(2*params[3])


gstar = 106.75
# gamma = 0.2222
D = 0.4444
A = 0.1990
# lamda = 0.0792
lamda = 0.0396
T0_Tc = 1./(np.sqrt(2))

params = (gstar, D, A, lamda, T0_Tc)

a0 = (params[0]*(np.pi**2)/30)

Tn_Tc = 0.8
ep = epsilon(Tn_Tc)
print 'Weak: ', ep

Tn_Tc = 0.86
ep = epsilon(Tn_Tc)
print 'Intermediate: ', ep