# From MPhys project by Andrew Hanna 2016-17

from scipy.optimize import fsolve
from scipy.integrate import odeint

from Toolbox import *

from matplotlib import pyplot as plt
import warnings



def get_Tminus(Tguess, Xiw):
    warnings.filterwarnings("error")
    try:
        Tm = fsolve(wminus3, Tguess, Xiw)
    except Warning:
        print 'Couldn''t find Tminus at Vwall = ',Xiw
        Tm = np.array([np.nan])
    return Tm


#Xiw_list = [0.1*v for v in range(7,10)]
#Tminus_list = []

Xiw_arr = np.linspace(0.7,0.95,12)
Tminus_arr = np.zeros_like(Xiw_arr)
vminus_arr = np.zeros_like(Xiw_arr)

Tguess = 0.8
for Xiw in Xiw_arr:
    print Xiw
    count = np.where(Xiw_arr == Xiw)
    Tminus = get_Tminus(Tguess, Xiw)
    print Tminus
    Tminus_arr[count] = Tminus[0]
    if not np.isnan(Tminus[0]):
        vminus_arr[count] = vminus(Tminus[0],Xiw)
    else:
        vminus_arr[count] = np.nan

print 'xiW_arr   ',Xiw_arr
print 'Tminus_arr',Tminus_arr
print 'vminus_arr',vminus_arr

#MBH Added a vminus array

#print 'phi_broken(Tn)', phi_broken(Tn_Tc)

#print 'Tplus', Tn_Tc
#print 'Tminus',Tminus_arr
#print 'vminus_arr', vminus_arr
#print 'vminus_plasma_arr', mu(Xiw_arr,vminus_arr)

#print 'alphaplus(Tn)', alphaplus(Tn_Tc)


#print 'alphaplus(Tminus)', alphaplus(Tminus_arr)
#print 'vCJ(Tminus)', vJouguet(Tminus_arr)

#print 'wminus1(Tminus)', wminus1(Tminus_arr)
#print 'wminus2(Tminus,Xiw)', wminus2(Tminus_arr,Xiw_arr)
#print 'vminus(Tminus,Xiw)', vminus(Tminus_arr,Xiw_arr)


nxi = 125
xi_end = 0.55

n_success = tuple(np.where(np.isfinite(Tminus_arr))[0])

for n in n_success:

    vminus_plasma0 = mu(Xiw_arr[n],vminus_arr[n])
    logT0 = np.log(Tminus_arr[n])

    print 'al_plus (Tminus = {}):  {}'.format(Tminus_arr[n],alphaplus(Tminus_arr[n]) )
    print 'vminus (Tminus = {}):   {}'.format(Tminus_arr[n],vminus_arr[n])
    print 'vJouguet (Tminus = {}): {}'.format(Tminus_arr[n],vJouguet(Tminus_arr[n]) )
    print 'cs(Tminus = {}):        {}'.format(Tminus_arr[n],cs(Tminus_arr[n]) )

    print n, Xiw_arr[n]
    Xiw = Xiw_arr[n]

    xi_arr = np.linspace(Xiw,xi_end,nxi)

    y0 = [vminus_plasma0, logT0]

# Integrate back from xiw

    sol = odeint(dy_dxi, y0, xi_arr)
    v = sol[:,0]
    logT = sol[:,1]
    # Also need to set w to constant behind place where v -> 0
    n_cs = np.max(np.where(v > 0.))
    logT[n_cs+1:] = logT[n_cs]
    # and v to zero behind v -> 0 place
    v[n_cs+1:] = 0

    if np.isfinite(Tminus_arr[n]):
        print 'Speed of trailing discontinuity behind wall: ',xi_arr[n_cs]

    xi2_arr = np.linspace(1, Xiw, num=14, endpoint=True)

    xi3_arr = np.concatenate((xi2_arr, xi_arr))

    vplus_arr = np.zeros_like(xi2_arr)
    vplus_arr.fill(0)

    v_arr = np.concatenate((vplus_arr,v))

    logTplus_arr = np.zeros_like(xi2_arr)
    logTplus_arr.fill(np.log(Tn_Tc))
#logTplus_arr.fill(-0.1443509)

    logT_arr = np.concatenate((logTplus_arr,logT))

    plt.figure(1)
    plt.xlabel('Xi')
    plt.ylabel('v')
    plt.plot(xi3_arr,v_arr)

    plt.figure(2)
    plt.xlabel('Xi')
    plt.ylabel('Wminus1(T)/Wminus(Tn)')
    plt.plot(xi3_arr, wminus(np.exp(logT_arr))/wminus(Tn_Tc))
#    plt.plot(xi3_arr, np.exp(logT_arr))

plt.show()

#plt.figure(1)
#plt.xlabel('Xiw')
#plt.ylabel('Tminus(Xiw)/Tn_Tc')
#plt.plot(Xiw_arr, Tminus_arr/Tn_Tc -1)

#plt.figure(2)
#plt.xlabel('Xiw')
#plt.ylabel('alphaplus(Tminus)/alphaplus(Tn_Tc)')
#plt.plot(Xiw_arr, alphaplus(Tminus_arr)/alphaplus(Tn_Tc) -1)

#plt.figure(3)
#plt.xlabel('Xiw')
#plt.ylabel('wminus1(Tminus)/wminus1(Tn_Tc)')
#plt.plot(Xiw_arr, wminus1(Tminus_arr)/wminus1(Tn_Tc) - 1)

#plt.figure(4)
#plt.xlabel('Xiw')
#plt.ylabel('wminus2(Tminus)/wminus2(Tn_Tc)')
#plt.plot(Xiw_arr, wminus2(Tminus_arr, Xiw_arr)/wminus2(Tn_Tc, Xiw_arr) - 1)

#plt.figure(5)
#plt.xlabel('Xiw')
#plt.ylabel('vminus(Tminus)/vminus(Tn_Tc)')
#plt.plot(Xiw_arr, vminus(Tminus_arr, Xiw_arr)/vminus(Tn_Tc, Xiw_arr) -1)

#plt.show()
