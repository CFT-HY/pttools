#!/usr/bin/env python
# Compare SSM prediction with data
# Creates and plots velocity and GW power spectra from SSM

from __future__ import absolute_import, division, print_function

import sys
# Add path for pttools
sys.path.append('../../../')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import bubble as b
import gw_code.ssmtools as ssm

print('Importing {}'.format(ssm.__file__))
print('Importing {}'.format(b.__file__))

# Set up plotting
# LaTeX can cause problems if system not configured correctly
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

font_size = 20
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams.update({'lines.linewidth': 1.5})
mpl.rcParams.update({'axes.linewidth': 2.0})
mpl.rcParams.update({'axes.labelsize': font_size})
mpl.rcParams.update({'xtick.labelsize': font_size})
mpl.rcParams.update({'ytick.labelsize': font_size})
# but make legend smaller
mpl.rcParams.update({'legend.fontsize': 14})

colour_list = ['b', 'r', 'g']

# Number of points used in the numerical calculations (n_z, n_xi, n_t)
# z - wavenumber space, xi - r/t space, t - time for size distribution integration
Np_list = [[1000, 2000, 200], [2500, 5000, 500], [5000, 10000, 1000]]  

nuc_type = 'simultaneous'
nuc_args = (1.,)
# Simultaneous is relevant for comparison to num sims
# Or:
# nuc_type = 'exponential'
# nuc_args = (1,)

zmin = 0.2   # Minimum z = k.R* array value
zmax = 1000  # Maximum z = k.R* array value


nz_string = 'nz'
for r in range(len(Np_list)):
    nz = Np_list[r][0]
    nz_string += '{}k'.format(nz // 1000)

nuc_string = nuc_type[0:3] + '_'
for n in range(len(nuc_args)):
    nuc_string += str(nuc_args[n]) + '_'

nt_string = '_nT{}'.format(Np_list[0][2])

file_type = 'pdf'

mdp = 'model_data/'  # model data path
gdp = 'graphs/'  # graph path

# All run parameters
    
vw_weak_list = [0.92, 0.80, 0.68, 0.56, 0.44]
vw_inter_list = [0.92, 0.72, 0.44]
# NB prace runs did not include intermediate 0.80, 0.56

eta_weak_list = [0.19, 0.35, 0.51, 0.59, 0.93]
eta_inter_list = [0.17, 0.40, 0.62]

alpha_weak = 0.0046
alpha_inter = 0.050

t_weak_list = [1210, 1380, 1630, 1860, 2520]
t_inter_list = [1180, 1480, 2650]

dt_weak_list = [0.08*t for t in t_weak_list]
dt_inter_list = [0.08*t for t in t_inter_list]
dt_inter_list[1]=0.05*t_inter_list[1] # dx=1

step_weak_list = [int(round(dt/20)*20) for dt in dt_weak_list]
step_inter_list = [int(round(dt/20)*20) for dt in dt_inter_list]

# Files used in prace paper 2017
#

dir_weak_list = ['results-weak-scaled_etatilde0.19_v0.92_dx2/',
        'results-weak-scaled_etatilde0.35_v0.80_dx2/',
        'results-weak-scaled_etatilde0.51_v0.68_dx2/',
        'results-weak-scaled_etatilde0.59_v0.56_dx2/',
        'results-weak-scaled_etatilde0.93_v0.44_dx2/'          
    ]
dir_inter_list = ['results-intermediate-scaled_etatilde0.17_v0.92_dx2/',
        'results-intermediate-scaled_etatilde0.40_v0.72_dx1/',
        'results-intermediate-scaled_etatilde0.62_v0.44_dx2/'
    ]

path_head = 'fluidprofiles/'
path_list_all = ['weak/', 'intermediate/']
file_pattern = 'data-extracted.{:05d}.txt'

vw_list_all = [vw_weak_list, vw_inter_list]
alpha_list_all = [alpha_weak, alpha_inter]
step_list_all = [step_weak_list, step_inter_list]
dir_list_all = [dir_weak_list, dir_inter_list]



def plot_ps(z_list, pow_list, ps_type, ax_limits='weak', 
            leg_list=None, col_list=None, ls_list=None, fig=None):
    # Plots a list of power spectra, with axis limits appropriate to prace runs
    # reurns a figure handle
    
    if col_list is None:
        col_list = ['b']*len(z_list)
        
    if ls_list is None:
        ls_list = ['-']*len(z_list)

    if fig is None:
        fig = plt.figure(figsize=[8, 4])
        ax = plt.gca()

    p_min, p_max = get_yaxis_limits(ps_type, ax_limits)

    power = []

    if leg_list is None:
        for z, pow, col, ls in zip(z_list, pow_list, col_list, ls_list):
            ax.loglog(z, pow, color=col, linestyle=ls)
            power.append(np.trapz(pow/z, z))
    else:
        for z, pow, leg, col, ls in zip(z_list, pow_list, leg_list, col_list, ls_list):
            ax.loglog(z, pow, color=col, linestyle=ls, label=leg)
            power.append(np.trapz(pow/z, z))

    # Pretty graphs
    ax.grid(True)
    ax.set_xlabel(r'$kR_*$')
    if ps_type == 'gw':
        ax.set_ylabel(r'$(H_{\rm n}R_*)^{-1}\mathcal{P}^\prime_{\rm ' + ps_type + '}(kR_*)$')
    else:
        ax.set_ylabel(r'$\mathcal{P}_{\rm ' + ps_type + '}(kR_*)$')
    ax.set_ylim([p_min, p_max])
    ax.set_xlim([zmin, zmax])
    if leg_list is not None:
        plt.legend(loc='best')
    plt.tight_layout()
    
    return fig


def generate_ps(vw, alpha, method='e_conserving', v_xi_file=None, save_ids=[None, None], show=True):    
    """ Generates, plots velocity and GW power as functions of $kR_*$.
    Saves power spectra in files pow_v_*, pow_gw_*...<string>.txt if save_id[0]=string.
    Saves plots in files pow_v_*, pow_gw_*...<string>.pdf if save_id[1]=string.
    Shows plots if show=True
    Returns <V^2> and Omgw divided by (Ht.HR*).
    """

    Np = Np_list[-1]
    col = colour_list[0]

    if alpha < 0.01:
        strength = 'weak'
    elif alpha < 0.1:  # intermediate transition
        strength = 'inter'
    else:
        sys.stderr.write('generate_ps: warning: alpha > 0.1, taking strength = strong\n')
        strength = 'strong'

# Generate power spectra
    z = np.logspace(np.log10(zmin), np.log10(zmax), Np[0])

    # Array using the minimum & maximum values set earlier, with Np[0] number of points
    print("vw = ", vw, "alpha = ", alpha, "Np = ", Np)

    sd_v = ssm.spec_den_v(z, [vw, alpha, nuc_type, nuc_args], Np[1:], method=method)
    pow_v = ssm.pow_spec(z,sd_v)
    V2_pow_v = np.trapz(pow_v/z, z)

    if v_xi_file is not None:
        sd_v2 = ssm.spec_den_v(z, [vw,alpha,nuc_type,nuc_args], Np[1:], v_xi_file, method=method)
        pow_v2 = ssm.pow_spec(z,sd_v2)
        V2_pow_v = np.trapz(pow_v2/z, z)

    sd_gw, y = ssm.spec_den_gw_scaled(z, sd_v)
    pow_gw = ssm.pow_spec(y, sd_gw)
    gw_power = np.trapz(pow_gw/y, y)    

    if v_xi_file is not None:
        sd_gw2, y = ssm.spec_den_gw_scaled(z,sd_v2)
        pow_gw2 = ssm.pow_spec(y, sd_gw2)
        gw_power = np.trapz(pow_gw2/y, y)    
        

# Now for some plotting if requested
    if save_ids[1] is not None or show:
        f1 = plt.figure(figsize=[8, 4])
        ax_v = plt.gca()
        
        f2 = plt.figure(figsize=[8, 4])
        ax_gw = plt.gca()
        
        ax_gw.loglog(y, pow_gw, color=col)
        ax_v.loglog(z, pow_v, color=col)
        
        if v_xi_file is not None:    
            ax_v.loglog(z, pow_v2, color=col, linestyle='--')
            ax_gw.loglog(y, pow_gw2, color=col, linestyle='--')
        
        
        inter_flag = (abs(b.cs0 - vw) < 0.05) # Due intermediate power law
        plot_guide_power_laws_prace(f1, f2, z, pow_v, y, pow_gw, inter_flag=inter_flag)
        
        # Pretty graph 1
        pv_min, pv_max = get_yaxis_limits('v',strength)
        ax_v.grid(True)
        ax_v.set_xlabel(r'$kR_*$')
        ax_v.set_ylabel(r'$\mathcal{P}_{\rm v}(kR_*)$')
        ax_v.set_ylim([pv_min, pv_max])
        ax_v.set_xlim([zmin, zmax])
        f1.tight_layout()
        
        # Pretty graph 2
        pgw_min, pgw_max = get_yaxis_limits('gw',strength)
        ax_gw.grid(True)
        ax_gw.set_xlabel(r'$kR_*$')
        ax_gw.set_ylabel(r'$\Omega^\prime_{\rm gw}(kR_*)/(H_{\rm n}R_*)$')
        ax_gw.set_ylim([pgw_min, pgw_max])
        ax_gw.set_xlim([zmin, zmax])
        f2.tight_layout()

# Now some saving if requested
    if save_ids[0] is not None or save_ids[1] is not None:
        nz_string = 'nz{}k'.format(Np[0] // 1000)
        nx_string = '_nx{}k'.format(Np[1] // 1000)
        nT_string = '_nT{}-'.format(Np[2])
        file_suffix = "vw{:3.2f}alpha{}_".format(vw, alpha) + nuc_string \
                        + nz_string + nx_string + nT_string

    if save_ids[0] is not None:
        data_file_suffix = file_suffix + save_ids[0] + '.txt'

        if v_xi_file is None:
            np.savetxt(mdp + 'pow_v_' + data_file_suffix, 
                       np.stack((z,pow_v),axis=-1), fmt='%.18e %.18e')
            np.savetxt(mdp + 'pow_gw_' + data_file_suffix, 
                       np.stack((y,pow_gw),axis=-1), fmt='%.18e %.18e')
        else:
            np.savetxt(mdp + 'pow_v_' + data_file_suffix, 
                       np.stack((z,pow_v,pow_v2),axis=-1), fmt='%.18e %.18e %.18e')
            np.savetxt(mdp + 'pow_gw_' + data_file_suffix, 
                       np.stack((y,pow_gw,pow_gw2),axis=-1), fmt='%.18e %.18e %.18e')
            
    if save_ids[1] is not None:
        graph_file_suffix = file_suffix + save_ids[1] + '.pdf'
        f1.savefig(gdp + "pow_v_" + graph_file_suffix)
        f2.savefig(gdp + "pow_gw_" + graph_file_suffix)

    # Now some diagnostic comparisons between real space <v^2> and Fourier space already calculated
    v_ip, w_ip, xi = b.fluid_shell(vw, alpha)
    Ubarf2 = b.ubarf_squared(v_ip, w_ip, xi, vw)

    print("vw = {}, alpha = {}, nucleation = {}".format(vw, alpha, nuc_string))
    print("<v^2> =                      ", V2_pow_v)
    print("Ubarf2 (1 bubble)            ", Ubarf2)
    print("Ratio <v^2>/Ubarf2           ", V2_pow_v/Ubarf2)
    print("gw power (scaled):           ", gw_power)

    if show:
        plt.show()
        
    return V2_pow_v, gw_power
# end function generate_ps

    
def all_generate_ps_prace(save_ids=['', ''], show=True):
    """Generate power spectra with Prace17 SSM parameters. 
    Save data files and graphs.
    Returns U-bar-f^2 and GW power as tuple of lists."""
   
    method = 'e_conserving'
    
    v2_list = []
    Omgw_scaled_list = []
    
    for vw_list, alpha, step_list, path, dir_list in zip(vw_list_all, alpha_list_all, 
                                                   step_list_all, path_list_all, dir_list_all):
        for vw, step, dir in zip(vw_list, step_list, dir_list):
            v_xi_file = path_head + path + dir + file_pattern.format(step)
            v2, Omgw = generate_ps(vw, alpha, method, v_xi_file, save_ids, show)
            v2_list.append(v2)
            Omgw_scaled_list.append(Omgw)
    
    return v2_list, Omgw_scaled_list
   
    
def get_yaxis_limits(ps_type,strength='weak'):
    if strength == 'weak':     
        if ps_type=='v':
            p_min = 1e-8
            p_max = 1e-3
        elif ps_type=='gw':
            p_min = 1e-16
            p_max = 1e-8
        else:
            p_min = 1e-8
            p_max = 1e-3
    elif strength == 'inter':  
        if ps_type=='v':
            p_min = 1e-7
            p_max = 1e-2
        elif ps_type=='gw':
            p_min = 1e-12
            p_max = 1e-4
        else:
            p_min = 1e-7
            p_max = 1e-2
    elif strength == 'strong':  
        if ps_type=='v':
            p_min = 1e-5
            p_max = 1
        elif ps_type=='gw':
            p_min = 1e-8
            p_max = 1
        else:
            p_min = 1e-5
            p_max = 1e-1
    else:
        sys.stderr.write('get_yaxis_limits: warning: strength = [ *weak | inter | strong]')
        if ps_type=='v':
            p_min = 1e-8
            p_max = 1e-3
        elif ps_type=='gw':
            p_min = 1e-16
            p_max = 1e-8
        else:
            p_min = 1e-8
            p_max = 1e-3
    
    return p_min, p_max
    

# Functions for plotting guide power laws on graphs

def get_ymax_location(x, y):
    # Returns x, y coordinates of maximum of array y
    ymax = max(y)
    xmax = x[np.where(y == ymax)][0]
    return np.array([xmax, ymax])

    
def plot_guide_power_law_prace(ax, x, y, n, position, shifts=None ):
    # Wrapper for plot_guide_power_law, with power laws and line 
    # shifts appropriate for velocity and GW spectra of prace runs
    if shifts==None:
        if position=='high':
            line_shift = [2, 1]
            txt_shift = [1.25, 1]
            xloglen = 1
        elif position=='med':
            line_shift = [0.5, 1.2]
            txt_shift = [0.5, 0.7]
            xloglen = -0.8
        elif position=='low':
            line_shift=[0.25, 0.25]
            txt_shift=[0.5, 0.5]
            xloglen = -1
        else:
            sys.exit('plot_guide_power_law_prace: error: position not recognised')
    else:
        line_shift = shifts[0]
        txt_shift = shifts[1]
        if position=='high':
            xloglen = 1
        elif position=='med':
            xloglen = -0.8
        elif position=='low':
            xloglen = -1
        else:
            sys.exit('plot_guide_power_law_prace: error: position not recognised')

    max_loc = get_ymax_location(x, y)

    power_law_loc = max_loc * np.array(line_shift)
    plot_guide_power_law(ax, power_law_loc, n, xloglen=xloglen, 
                         txt=r'$k^{{{:}}}$'.format(n), txt_shift=txt_shift)

    return power_law_loc

    
def plot_guide_power_law(ax, loc, power, xloglen=1, txt='', txt_shift=[1, 1], col='k', ls='-'):
    # Plot a guide power law going through loc[0], loc[1] with index power
    # Optional annotation at (loc[0]*txt_shift[0], loc[1]*txt_shift[1])
    # Returns the points in two arrays (is this the best thing?)
    xp = loc[0]
    yp = loc[1]
    x_guide = np.logspace(np.log10(xp), np.log10(xp) + xloglen, 2)
    y_guide = yp*(x_guide/xp)**power
    ax.loglog(x_guide, y_guide, color=col, linestyle=ls)

    if txt != '':
        txt_loc = loc * np.array(txt_shift)
        ax.text(txt_loc[0], txt_loc[1], txt, fontsize=16)

    return x_guide, y_guide


def plot_guide_power_laws_ssm(f, z, pow, ps_type='v', inter_flag=False):
    # Plot guide power laws (assumes params all same for list)
    # Shifts designed for simulataneous nucleation lines
    x_high = 10
    x_low  = 3
    if ps_type == 'v':
        n_lo, n_med, n_hi = 5, 1, -1
        shifts_hi = [[2,1],[1.5,1]]
        shifts_lo = [[0.5,0.25],[0.5,0.15]]
    elif ps_type == 'gw':
        n_lo, n_med, n_hi = 9, 1, -3
        shifts_hi = None
        shifts_lo = [[0.6,0.25],[0.5,0.08]]
    else:
        n_lo, n_med, n_hi = tuple(ps_type)

    sys.stderr.write('plot_guide_power_laws_ssm: Plotting guide power laws\n')

    high_peak  = np.where(z > x_high)
    plot_guide_power_law_prace(f.axes[0], z[high_peak], pow[high_peak], n_hi, 'high', 
                                   shifts=shifts_hi)

    if inter_flag:
        # intermediate power law to be plotted
        plot_guide_power_law_prace(f.axes[0], z[high_peak], pow[high_peak], n_med, 'med')

    low_peak = np.where(z < x_low)
    plot_guide_power_law_prace(f.axes[0], z[low_peak], pow[low_peak], n_lo, 'low', 
                               shifts=shifts_lo)

    return f


def plot_guide_power_laws_prace(f_v, f_gw, z, pow_v, y, pow_gw, 
                                np_lo=[5, 9], inter_flag=False):
    # Plot guide power laws (assumes params all same for list)
    # Shifts designed for simulataneous nucleation lines
    x_high = 10
    x_low  = 2
    [nv_lo, ngw_lo] = np_lo
    sys.stderr.write('Plotting guide power laws\n')
    high_peak_v  = np.where(z > x_high)
    high_peak_gw = np.where(y > x_high)
    plot_guide_power_law_prace(f_v.axes[0], z[high_peak_v], pow_v[high_peak_v], -1, 'high', 
                                   shifts=[[2,1],[2.1,0.6]])
    plot_guide_power_law_prace(f_gw.axes[0], y[high_peak_gw], pow_gw[high_peak_gw], -3, 'high')

    if inter_flag:
        # intermediate power law to be plotted
        plot_guide_power_law_prace(f_v.axes[0], z[high_peak_v], pow_v[high_peak_v], 1, 'med')
        plot_guide_power_law_prace(f_gw.axes[0], y[high_peak_gw], pow_gw[high_peak_gw], 1, 'med',
                                   shifts=[[0.5, 1.5],[0.5, 2]])

    low_peak_v = np.where(z < x_low)
    low_peak_gw = np.where(y < x_low)
    plot_guide_power_law_prace(f_v.axes[0], z[low_peak_v], pow_v[low_peak_v], nv_lo, 'low', 
                               shifts=[[0.5,0.25],[0.5,0.15]])
#    plot_guide_power_law_prace(f_gw.axes[0], y[low_peak_gw], pow_gw[low_peak_gw], ngw_lo, 'low',
#                               shifts=[[0.4,0.5],[0.5,0.5]])
    plot_guide_power_law_prace(f_gw.axes[0], y[low_peak_gw], pow_gw[low_peak_gw], ngw_lo, 'low',
                               shifts=[[0.6,0.25],[0.5,0.08]])

    return f_v, f_gw

    
