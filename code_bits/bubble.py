#!/usr/bin/env python
#
# Functions for calculating fluid profile around expanding Higgs-phase bubble.
# See Espinosa et al 2010
#
# Mudhahir Al-Ajmi and Mark Hindmarsh 2015-16

import sys, os
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# Should think about true cs
cs=1/np.sqrt(3) # speed of sound

def vPlus(vm,ap, wallType):
#    Wall frame speed ahead of the wall
    X = vm + 1./(3*vm)
    if wallType == 'Deflagration' or wallType == 'Hybrid':
        b = -1.
    else:
        b = 1.
    return (0.5/(1+ap))*( X + b*np.sqrt(X**2 + 4.*ap**2 + (8./3.)*ap - (4./3.)) )

def vMinus(vp,ap):
#    Wall frame speed behind the wall
    vp2 = vp**2
    Y = vp2 + 1./3.
    Z = (Y - ap*(1. - vp2))
    X = (4./3.)*vp2
    if Z**2 < X:
      print "Error (Detonation): High value of Alpha_+"
      return False
#      sys.exit(1)
    else:
      return (0.5/vp)*( Z + np.sqrt( Z**2 - X ) )

def vShock(xis):
# Fluid velocity at a shock at xis.  No shock for xis < cs, so returns zero    
    return np.maximum(0.,(xis**2 - cs**2)/(xis*(1 - cs**2)) )

def lorentz(xi,v):
#    Lorentz Transformation of fluid speed v between local moving frame and plasma frame.
##    print "loretenz: xi.shape, v.shape", xi.shape, v.shape
    return (xi - v)/(1 - v*xi)

def dv_dxi_deflag(v1,x):
#    differential equation: dv/dxi  for deflgrations
    if v1 < vShock(x): # Can happen if you try to integrate beyond shock
        val = 0.       # Stops solution blowing up
    else:
        val = (2./x)*v1*(1.-v1**2)*(1./(1-x*v1))*(1./(lorentz(x,v1)**2/cs**2 - 1))
    return val

def dv_dxi_deton(v1,x):
    #    differential equation: dv/dxi  for detonations and hybrids (integrate backwards from wall)
    val = (2./x)*v1*(1.-v1**2)*(1./(1-x*v1))*(1./(lorentz(x,v1)**2/cs**2 - 1))
    return val

def v_just_behind(x,v,dx):
    # Fluid velocity one extra space step behind wall, arranged so that dv_dxi_deton guaranteed positive
    dv = np.sqrt(4.*dx*v*(1-v*v)*(x-v)/(1-x*x))
    return v-dv

def min_speed_deton(al_p):
    # Minimum speed for a detonation
    return (cs/(1 + al_p))*(1 + np.sqrt(al_p*(2. + 3.*al_p)))

def max_speed_deflag(al_p):
    # Maximum speed for a deflagration
    vm=cs
    return 1/(3*vPlus(vm, al_p, 'Deflagration'))

def enthalpyEn(v1,x):
    # Calculating enthalpy
    lf = LorentzFactor2(v1)
    lb = lorentz(x,v1)
    return (1.+1./cs**2)*lf*lb

def LorentzFactor2(v1):
    #Calculating squaare of gamma 
    gamma = (1.)/(1. - v1**2)
    return gamma

def xvariables(Npts, v_w):
    dxi = 1./Npts
    xi = np.linspace(0., 1., num=(Npts+1))
    v_sh = np.zeros(Npts+1)
    nWall = np.int(np.floor(v_w/dxi))
    ncs = np.int(np.floor(cs/dxi))
    v_sh[ncs:] = vShock(xi[ncs:])
    xi_ahead = xi[nWall:]
    xi_behind = xi[nWall-1:ncs:-1]
    return dxi, xi, v_sh, nWall, ncs, xi_ahead, xi_behind

def derived_parameters(v_w,Npts):
    dxi = 1./Npts
    nWall = np.int(np.floor(v_w/dxi))
    return dxi, nWall

#def wallVariables(Alpha_f, vWall):
def wallVariables(Alpha_f, wallType, vWall):
    if vWall <= 1:
#print "max_speed_deflag(Alpha_f)= ", max_speed_deflag(Alpha_f)
#     if vWall < max_speed_deflag(Alpha_f) and vWall <= cs and Alpha_f <= 1/3.:
        if wallType == 'Deflagration':
             vfm_w = vWall                            # Fluid velocity just behind the wall in wall frame (v-)
             vfm_p = lorentz(vWall,vfm_w)             # Fluid velocity just behind the wall in plasma frame
             vfp_w = vPlus(vWall,Alpha_f,wallType)    # Fluid velocity just ahead of the wall in wall frame (v+)
             vfp_p = lorentz(vWall,vfp_w)             # Fluid velocity just ahead of the wall in plasma frame
    #    elif vWall <= max_speed_deflag(Alpha_f) and vWall > cs and Alpha_f <= 1/3.:
        elif wallType == 'Hybrid':
             vfm_w = cs                               # Fluid velocity just behind the wall in plasma frame (hybrid)
             vfm_p = lorentz(vWall,vfm_w)             # Fluid velocity just behind the wall in plasma frame    
             vfp_w = vPlus(cs,Alpha_f,wallType)       # Fluid velocity just ahead of the wall in wall frame (v+)
             vfp_p = lorentz(vWall,vfp_w)             # Fluid velocity just ahead of the wall in plasma frame
    #     elif not (vWall < max_speed_deflag(Alpha_f) and vWall <= cs) and not (vWall < max_speed_deflag(Alpha_f) and vWall > cs) and vWall > max_speed_deflag(Alpha_f):
    #     elif vWall > max_speed_deflag(Alpha_f) or Alpha_f > 1/3.:
    #     else:
        elif wallType == 'Detonation':
             vfm_w = vMinus(vWall,Alpha_f)            # Fluid velocity just behind the wall in wall frame (v-)
             vfm_p = lorentz(vWall,vfm_w)             # Fluid velocity just behind the wall in plasma frame
             vfp_w = vWall                            # Fluid velocity just ahead of the wall in wall frame (v+)
             vfp_p = lorentz(vWall,vfp_w)             # Fluid velocity just ahead of the wall in plasma frame
        else:
          print "wallVariables: error: wallType wrong or unset"
          sys.exit(1)
    else:
        print "wallVariables: error: vWall > 1"
#    return wallType, vfp_w, vfm_w, vfp_p, vfm_p
    return vfp_w, vfm_w, vfp_p, vfm_p

#def fluid_shell(v_w, al_p, Npts):
#    # Replaced by velocity(v_w, al_p, wallType, Npts) Jan 2017
#    vFluid = np.zeros([Npts+1,1]) # Could make size [Npts,len(v_w)] 
#    wallType, vfp_w, vfm_w, vfp_p, vfm_p = wallVariables(al_p,  v_w)
#    dxi, xi, v_sh, nWall, ncs, xi_ahead, xi_behind = xvariables(Npts, v_w)   # initiating x-axis variables
#
##   Calculating the fluid velocity
#    vFluid[nWall:] = scipy.integrate.odeint(dv_dxi_deflag,vfp_p,xi_ahead)#,mxstep=5000000)
#    vFluid[nWall-1:ncs:-1] = scipy.integrate.odeint(dv_dxi_deton,
#                                        v_just_behind(xi[nWall],vfm_p,dxi),xi_behind)#,mxstep=5000000)
#    if not (wallType=="Detonation"):
#     for n in range(nWall,Npts):
#        if vFluid[n] < v_sh[n]:
#          nShock = n
#          break
#    else: nShock = nWall
#        # Set fluid velocity to zero in front of the shock (integration isn't correct in front of shock)
#
#    vFluid[nShock:] = 0.0
#
###    print "nShock= ", nShock
###    print "xi[nShock]", xi[nShock]
###    print "Shock speed:", xi[nShock]
#
#    return vFluid

def velocity(v_w, al_p, wallType, Npts):
    
    vFluid = np.zeros([Npts+1,1]) # Could make size [Npts,len(v_w)]
    vfp_w, vfm_w, vfp_p, vfm_p = wallVariables(al_p,wallType, v_w)
    dxi, xi, v_sh, nWall, ncs, xi_ahead, xi_behind = xvariables(Npts, v_w)   # initiating x-axis variables
    
    #   Calculating the fluid velocity
    vFluid[nWall:] = scipy.integrate.odeint(dv_dxi_deflag,vfp_p,xi_ahead)#,mxstep=5000000)
    vFluid[nWall-1:ncs:-1] = scipy.integrate.odeint(dv_dxi_deton,
                                                    v_just_behind(xi[nWall],vfm_p,dxi),xi_behind)#,mxstep=5000000)
    if not (wallType=="Detonation"):
        for n in range(nWall,Npts):
            if vFluid[n] < v_sh[n]:
                nShock = n
                break
    else: nShock = nWall

# Set fluid velocity to zero in front of the shock (integration isn't correct in front of shock)
    vFluid[nShock:] = 0.0
    
    return vFluid

def enthalpy(vWall, Alpha_f, wallType, Np, v_f):
# Calculate enthalpy by integrating velocity, and normalising to enthalpy at T nucleation

    enthalpyY = np.zeros([Np+1,1]) # Could make size [Npts,len(v_w)]
    vfp_w, vfm_w, vfp_p, vfm_p = wallVariables(Alpha_f, wallType, vWall)
    dxi, xi, v_sh, nWall, ncs, xi_ahead, xi_behind = xvariables(Np, vWall)   # initiating x-axis variables

    if not (wallType=="Detonation"):
     for n in range(nWall,Np):
        if v_f[n] < v_sh[n]:
          nShock = n
          break
    else: nShock = nWall

    alp_s=0                               # in the plasma
    if not (wallType=="Detonation"):
     vfm_s = xi[nShock]                   # Shock wall speed
     vfp_s = vPlus(vfm_s,alp_s, wallType)  # Fluid velocity just ahead of the shock wall in the shock wall frame
    else:
     vfp_s = xi[nShock]                   # Shock wall speed
     vfm_s = vMinus(vfp_s,alp_s)          # Fluid velocity just ahead of the shock wall in the shock wall frame

    rw=(vfm_w*(1-vfp_w**2))/((vfp_w)*(1-vfm_w**2))
    rs=(3*vfm_s-vfp_s)/(3*vfp_s-vfm_s)

    en_exp_integrand = enthalpyEn(v_f[:nShock,0], xi[:nShock])
    en_exp = scipy.integrate.cumtrapz(en_exp_integrand,v_f[:nShock,0])
    enthalpyY[1:nShock,0] = np.exp(en_exp)
    shock_Enthalpy=np.exp(en_exp)[len(en_exp)-1]

    # Now normalise to enthalpy at large distrance (enthalpy at T_nucleation)

    if wallType == "Deflagration":
      enthalpyY[:nWall-1,0] *= enthalpyY[nWall,0]/(enthalpyY[nWall-1,0]*rw) 
      enthalpyY[nShock:,0] = shock_Enthalpy/(rs)
    if wallType == "Hybrid":
      enthalpyY[nShock:,0] = shock_Enthalpy/(rs)
      enthalpyY[:nWall-1,0] *= enthalpyY[nWall,0]/(enthalpyY[nWall-1,0]*rw)
    if wallType == "Detonation":
      enthalpyY[nShock:,0] = shock_Enthalpy*rw

    enthalpyNY = enthalpyY[nShock,0]
    enthalpyY=enthalpyY/enthalpyY[nShock,0]

    return enthalpyY

def fluid_shell(vWall, Alpha_f, wallType, Np):
    # New routine to return v_f, enthalpy pair
    v_f = velocity(vWall, Alpha_f, wallType,  Np)
    enthalp = enthalpy(vWall, Alpha_f, wallType, Np, v_f)
    return v_f, enthalp

def alphaN(vWall_,ap_, wallType_,Np_):
# Calculates alpha_N (relative latent heat outside bubble) from alpha_plus (ap_)
    dxi, nWall = derived_parameters(vWall_,Np_)
    v_ = velocity(vWall_, ap_, wallType_, Np_)
    w_ = enthalpy(vWall_, ap_, wallType_, Np_, v_)
    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalised to 1 at large xi
    return w_[nWall]*ap_

def alphaPlusMaxDetonation(vWall_):
    # Maximum allowed value of alpha_+ for a detonation with wall speed vWall_
    a= 3*(1-vWall_**2)
    b= (1-np.sqrt(3)*vWall_)**2
    b[np.where(vWall_ < 1./np.sqrt(3))] = 0.0
    return b/a

def alphaPlusMinHybrid(vWall_):
    # Minimum allowed value of alpha_+ for a hybrid with wall speed vWall_
    b= (1-np.sqrt(3)*vWall_)**2
    c= (9*vWall_**2-1)
    b[np.where(vWall_ < 1./np.sqrt(3))] = 0.0
    return b/c

def find_alphaPlus(aN,vWall):
#
# define function fun(x) = alphaN(x,vWall)  - alN
#
# find the root, i.e. solution fun(x) = 0
#
# return x
    return 0.0
