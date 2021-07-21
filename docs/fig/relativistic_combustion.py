# Import modules
import sys

sys.path.insert(0, "/Users/dacuttin/Projects/pttools/bubble/")
import bubble as b
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

## setup latex plotting
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
## make font size bigger
matplotlib.rcParams.update({'font.size': 28})
## but make legend smaller
matplotlib.rcParams.update({'legend.fontsize': 14})
## change line thickness
matplotlib.rcParams.update({'lines.linewidth': 1.75})
from scipy.interpolate import interp1d

import seaborn as sb
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm

fig, ax = plt.subplots(1, 3, figsize=(27, 9))

alpha = 0.5
vws = [0.44, 0.72, 0.92]
plot_cbar = [False, False, True]
viridisBig = cm.get_cmap('autumn_r', 512)

newcolors = viridisBig(np.linspace(0, 1, 256))
newcolors[0] = matplotlib.colors.to_rgba('white', alpha=0)
newcmp = ListedColormap(newcolors)

Np = 5000

label_s = [r'subsonic deflagration' + '\n' + r'$v_\mathrm{w} \leq c_s$',
           r'supersonic deflagration' + '\n' + r'$c_s<v_\mathrm{w} < c_\mathrm{J}$',
           r'detonation' + '\n' + r'$c_s<c_\mathrm{J}\leq v_\mathrm{w}$']
print(label_s)
for i in range(len(vws)):

    v_f, enthalp, xi = b.fluid_shell(vws[i], alpha, Np)
    n_wall = b.find_v_index(xi, vws[i])
    v_fluid = interp1d(xi, v_f, fill_value=0, bounds_error=False)

    xvalues = np.linspace(-1.5 * xi[n_wall], 1.5 * xi[n_wall], num=4000)
    yvalues = np.linspace(-1.5 * xi[n_wall], 1.5 * xi[n_wall], num=4000)

    xxgrid, yygrid = np.meshgrid(xvalues, yvalues)

    fluid_grid = v_fluid(np.sqrt((xxgrid) * (xxgrid) + (yygrid) * (yygrid)))

    fluid_grid = fluid_grid / np.max(fluid_grid)

    arrow_width = 0.03 * xi[n_wall]

    cs = ax[i].imshow(fluid_grid, cmap=newcmp,
                      extent=(-1.5 * xi[n_wall], 1.5 * xi[n_wall], -1.5 * xi[n_wall], 1.5 * xi[n_wall]),
                      interpolation='bilinear')
    circle = plt.Circle((0, 0), xi[n_wall], color='k', linewidth=4, fill=None)
    ax[i].arrow(0.75 * xi[n_wall], 0, 0.5 * xi[n_wall], 0, shape='full', width=arrow_width, edgecolor='k',
                facecolor='k')
    ax[i].arrow(0. * xi[n_wall], 0.75 * xi[n_wall], 0, 0.5 * xi[n_wall], shape='full', width=arrow_width, edgecolor='k',
                facecolor='k')
    ax[i].arrow(-0.75 * xi[n_wall], 0, -0.5 * xi[n_wall], 0, shape='full', width=arrow_width, edgecolor='k',
                facecolor='k')
    ax[i].arrow(0. * xi[n_wall], -0.75 * xi[n_wall], 0, -0.5 * xi[n_wall], shape='full', width=arrow_width,
                edgecolor='k', facecolor='k')
    ax[i].arrow((0.75 * xi[n_wall]) / np.sqrt(2), (0.75 * xi[n_wall]) / np.sqrt(2), (0.5 * xi[n_wall]) / np.sqrt(2),
                (0.5 * xi[n_wall]) / np.sqrt(2), shape='full', width=arrow_width, edgecolor='k', facecolor='k')
    ax[i].arrow(-(0.75 * xi[n_wall]) / np.sqrt(2), (0.75 * xi[n_wall]) / np.sqrt(2), -(0.5 * xi[n_wall]) / np.sqrt(2),
                (0.5 * xi[n_wall]) / np.sqrt(2), shape='full', width=arrow_width, edgecolor='k', facecolor='k')
    ax[i].arrow((0.75 * xi[n_wall]) / np.sqrt(2), -(0.75 * xi[n_wall]) / np.sqrt(2), (0.5 * xi[n_wall]) / np.sqrt(2),
                -(0.5 * xi[n_wall]) / np.sqrt(2), shape='full', width=arrow_width, edgecolor='k', facecolor='k')
    ax[i].arrow(-(0.75 * xi[n_wall]) / np.sqrt(2), -(0.75 * xi[n_wall]) / np.sqrt(2), -(0.5 * xi[n_wall]) / np.sqrt(2),
                -(0.5 * xi[n_wall]) / np.sqrt(2), shape='full', width=arrow_width, edgecolor='k', facecolor='k')

    if plot_cbar[i] == True:
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label(r'$v/v_\mathrm{peak}$')
    ax[i].add_artist(circle)
    ax[i].axis('off')

    ax[i].annotate(label_s[i], (0.51, -0.1), xycoords='axes fraction', ha='center', va='center', fontsize=30)

plt.show()
fig.savefig('plots/all_circle.pdf', bbox_inches='tight')
