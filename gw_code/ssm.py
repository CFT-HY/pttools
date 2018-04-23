#!/usr/bin/env python

from ssmtools import *
import matplotlib.pyplot as plt

#Inputs:
# Type='deflagration'#Type of transitions
#Valid inputs are: 'detonation','deflagration' & 'hybrid'
# beta=100 #Rate of bubble nucleation $\textcolor{red}{\beta}$

# Derived parameters:
# Rstar=(8.*np.pi)**(1./3.)*vw/beta #Average separation between bubbles
# Lf=Rstar #Scaling parameter $\textcolor{red}{L_\text{f}}$
# kmin=0.1/Lf #Minimum array value
# kmax=1000/Lf #Maximum array value
# k=np.logspace(np.log10(kmin),np.log10(kmax),Np)
#Array using the minimum & maximum values set earlier, with Np number of points

vw = 0.44       # Wall speed $\textcolor{red}{v_w}$
alpha = 0.0046  # Strength of interaction $\textcolor{red}{\alpha}$
wall_type = 'Calculate'  # Let code figure out the wall type

nz = 1000  # Number of points in kR* array
nxi = nz   # Number of points in xi array
nt = 200   # Number of points in kR* array
Np = [nz, nxi, nt] # List to pass as parameter

n=[2., ]  # Parameter for the bubble lifetime distribution - must be list or tuplee

zmin=0.1  # Minimum kR* array value
zmax=1000 # Maximum kR* array value
z=np.logspace(np.log10(zmin),np.log10(zmax),nz)

# Set up plotting
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=16)


# plt.figure(1,figsize=[8,5])
# plt.loglog(k*Lf,Mathcal_P_v(kmin,kmax,Lf,n,vw,alpha,beta,Np,Type))
# #Plots the Velocity power spectrum $\textcolor{red}{\mathcal{P}_v}$ as a function of $\textcolor{red}{kR_*}$
# plt.xlabel(r'$kR_*$')
# plt.ylabel(r'$\mathcal{P}_{\rm v}(kR_*)$')
# plt.tight_layout()


#Plots the GW power spectrum $\textcolor{red}{\mathcal{P}_{GW}}$ as a function of $\textcolor{red}{kR_*}$
plt.figure(1,figsize=[10,5])
ax = plt.gca()

plt.loglog(z, Mathcal_P_GW(z, vw, alpha, wall_type, 'simultaneous', [1.], Np), label='sim, 1')
plt.loglog(z, Mathcal_P_GW(z, vw, alpha, wall_type, 'simultaneous', [2.], Np), label='sim, 2')
plt.loglog(z, Mathcal_P_GW(z, vw, alpha, wall_type, 'exponential', [0.], Np), label='exp, 0')
plt.loglog(z, Mathcal_P_GW(z, vw, alpha, wall_type, 'exponential', [2.], Np), label='exp, 2')

plt.xlabel(r'$kR_*$')
plt.ylabel(r'$\Omega_{\rm gw}(kR_*)$')
ax.set_ylim([1e-16, 1e-8])
plt.legend()

plt.tight_layout()

plt.show()