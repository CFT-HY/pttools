"""
Entropy data generator (old reference)
======================================

Created on Wed Jun 23 15:53:31 2021

@author: hindmars
"""

# import sys
# sys.path.append('../pttools/')
# sys.path.append('../thermo/')

import pttools.bubble as b
import thermo.eos as eos
import numpy as np

# n_alpha = 100
# n_vw = 100
# alpha_list = np.logspace(-2.0,0.0,n_alpha)
# vw_list = np.linspace(0.005,0.995,n_vw)

n_alpha = 10
n_vw = 10
alpha_list = np.linspace(0.05, 0.95, n_alpha)
vw_list = np.linspace(0.05, 0.95, n_vw)

# g_bro = 20
# g_bro = eos.G_BRO_DEFAULT
g_bro = 120


def get_entropy_diff(v_wall, alpha, g_bro=eos.G_BRO_DEFAULT, n_xi=b.N_XI_DEFAULT):
    bg = eos.bag_model(alpha, g_bro=g_bro)

    v,w,xi = b.sound_shell_bag(v_wall, alpha, n_xi=n_xi)
    if not any(np.isnan(v)):
        # entropy change
        T = bg.T(w,b.phase(xi,v_wall))
        s = bg.s(T,b.phase(xi,v_wall))
        S_tot = np.trapz(s,xi**3)
        S_tot0 = np.trapz(s[-1]*np.ones_like(xi),xi**3)
        S_in0 = np.trapz(s[-1]*np.ones_like(xi)*b.phase(xi,v_wall),xi**3)
        dS_S0 = (S_tot - S_tot0)/S_in0
    else:
        dS_S0 = np.nan
    
    return dS_S0


def get_entropy_diff_arr(vw_list, alpha_list, g_bro=eos.G_BRO_DEFAULT):
    n_vw = len(vw_list)
    n_alpha = len(alpha_list)
    
    ds_list = []
    
    for alpha in alpha_list:
        for vw in vw_list:
            ds = get_entropy_diff(vw, alpha, g_bro)
            ds_list.append(ds)
            print('{:5.3f}, {:5.3f}: {:5.3f}'.format(vw,alpha, ds))

    ds_arr = np.array(ds_list).reshape(n_alpha, n_vw)

    return ds_arr, np.array(vw_list), np.array(alpha_list)


def get_pressure_diff(v_wall, alpha, g_bro=eos.G_BRO_DEFAULT, n_xi=b.N_XI_DEFAULT):
    bg = eos.bag_model(alpha, g_bro=g_bro)

    v,w,xi = b.sound_shell_bag(v_wall, alpha, n_xi=n_xi)
    if not any(np.isnan(v)):
        # pressure change
        T = bg.T(w,b.phase(xi,v_wall))
        p = bg.p(T, b.phase(xi,v_wall))
        p_nuc = p[xi > v_wall][-1]
        p_minus = p[xi < v_wall][-1]
#        print(p_plus, p_minus, w[-1])
        dp_wn = (p_minus - p_nuc)/w[-1]
    else:
        dp_wn = np.nan
        
    return dp_wn


def get_pressure_diff_arr(vw_list, alpha_list, g_bro=eos.G_BRO_DEFAULT):
    n_vw = len(vw_list)
    n_alpha = len(alpha_list)
    
    dp_list = []
    
    for alpha in alpha_list:
        for vw in vw_list:
            dp = get_pressure_diff(vw,alpha)
            dp_list.append(dp)
            print('{:5.3f}, {:5.3f}: {:5.3f}'.format(vw,alpha, dp))

    dp_arr = np.array(dp_list).reshape(n_alpha,n_vw)

    return dp_arr, np.array(vw_list), np.array(alpha_list)


def get_s_p_diffs_arr(vw_list, alpha_list, g_bro=eos.G_BRO_DEFAULT):
    n_vw = len(vw_list)
    n_alpha = len(alpha_list)
    
    dp_list = []
    ds_list = []

    for alpha in alpha_list:
        for vw in vw_list:
            dp = get_pressure_diff(vw,alpha)
            ds = get_entropy_diff(vw, alpha, g_bro)
            dp_list.append(dp)
            ds_list.append(ds)
            print('{:5.3f}, {:5.3f}: {:5.3f}, {:5.3f}'.format(vw,alpha, dp, ds))

    dp_arr = np.array(dp_list).reshape(n_alpha, n_vw)
    ds_arr = np.array(ds_list).reshape(n_alpha, n_vw)

    return dp_arr, ds_arr, np.array(vw_list), np.array(alpha_list)


dp_arr, ds_arr, vw_arr, alpha_arr = get_s_p_diffs_arr(vw_list, alpha_list, g_bro)


file_name_p = 'p_change_gbro{:3.0f}_g_sym{:3.0f}_nalpha_{}_nvw_{}.npz'.format(g_bro, eos.G_SYM_DEFAULT, n_alpha, n_vw)
np.savez(file_name_p, dp_arr, vw_arr, alpha_arr)

file_name_s = 's_change_gbro{:3.0f}_g_sym{:3.0f}_nalpha_{}_nvw_{}.npz'.format(g_bro, eos.G_SYM_DEFAULT, n_alpha, n_vw)
np.savez(file_name_s, ds_arr, vw_arr, alpha_arr)
