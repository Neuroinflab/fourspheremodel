from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec, cm
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import os
import parameters as params

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d',
                    default='results',
                    dest='results',
                    help='a path to the result directory')

args = parser.parse_args()

I = 1.0
dipole_list = params.dipole_list
P1s = []
name_list = []
radii = [params.brain_rad, params.csftop_rad,
         params.skull_rad, params.scalp_rad]

for dipole in dipole_list:
    src = np.array(dipole["src_pos"])
    snk = np.array(dipole["snk_pos"])
    P1s.append(np.array((src * I, -1 * snk * I)))
    name_list.append(dipole["name"])

scaling_k = 100.
ana_rad = np.load(os.path.join(args.results, "Analytical_rad.npz"))['phi_20']*scaling_k
ana_tan = np.load(os.path.join(args.results, "Analytical_tan.npz"))['phi_20']*scaling_k
ana_mix = np.load(os.path.join(args.results, "Analytical_mix.npz"))['phi_20']*scaling_k

num_rad = np.load(os.path.join(args.results, "Numerical_rad.npz"))['fem_20']*scaling_k
num_tan = np.load(os.path.join(args.results, "Numerical_tan.npz"))['fem_20']*scaling_k
num_mix = np.load(os.path.join(args.results, "Numerical_mix.npz"))['fem_20']*scaling_k

fem_list = [num_rad, num_tan, num_mix]
ana_list = [ana_rad, ana_tan, ana_mix]

fem_list = [ii.reshape(180, 180) for ii in fem_list]
ana_list = [ii.reshape(180, 180) for ii in ana_list]
error_list = [100.*np.abs((ii - jj)/np.max(jj)) for ii, jj in zip(fem_list, ana_list)] 
error_max = [np.max(ii) for ii in error_list]

abs_max_val = [np.max(np.abs(ii)) for ii in ana_list]
abs_max_range = [0.3 * ii for ii in abs_max_val]
print('Error max for rad, tan, mix: ', error_max)
print('Abs max for rad, tan, min', [np.max(np.abs(ii)) for ii in ana_list])


def set_axis(ax, letter):
    ax.text(-.20,
            0.9,
            letter,
            fontsize=12,
            weight='bold',
            transform=ax.transAxes)
    return ax


def set_axis_3d(ax, letter):
    ax.text(0.05,
            1.025,
            100.025,
            letter,
            fontsize=12,
            weight='bold',
            transform=ax.transAxes)
    return ax


def draw_diagram(ax):
    color_list = ['#ffcc99', '#e6e6e6', '#83caff', '#999999']
    for idx in range(4):  # Hack - circles are overlaid on each other
        c = plt.Circle((0, 0), radii[-(idx + 1)], color=color_list[idx])
        ax.add_artist(c)
    ax.arrow(-10, -10, 0, 1.8, head_width=.4, head_length=.4, fc='k', ec='k')
    ax.arrow(-10, -10, 1.8, 0, head_width=.4, head_length=.4, fc='k', ec='k')
    ax.arrow(-10, -10, -0.78, -0.78, head_width=.4, head_length=.33, fc='k', ec='k')
    ax.text(-11.5, -12, 'x', size=6.)
    ax.text(-7, -10.38, 'y', size=6.)
    ax.text(-10.28, -7, 'z', size=6.)
    return ax


def adjust_3d_axis(ax, ii):
    ax.set_xlim3d(-6.5, 6.5)
    ax.set_ylim3d(-6.5, 6.5)
    ax.set_zlim3d(0.3 - 6.5, 0.3 + 6.5)
    ax.view_init(10, 0)
    ax.set_aspect('equal')
    #ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(row_texts[ii], labelpad=0.1, rotation=90)
    ax.set_ylabel(sphere_xlabel[ii], labelpad=0.1)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])
    return ax


z_steps = 4
height_ratios = [1 for i in range(z_steps - 1)]
height_ratios.append(0.07)  # height of colorbar
fig = plt.figure(figsize=(12, 10))
#gs = gridspec.GridSpec(z_steps, 3, height_ratios=height_ratios, wspace=0.3, hspace=0.4)
gs = gridspec.GridSpec(z_steps, 4, height_ratios=height_ratios, wspace=0.3, hspace=0.4)

# for ii, P1, letter in zip([0, 1, 2], P1s, ['A', 'E', 'I']):
#     ax = plt.subplot(gs[ii, 0])
#     draw_diagram(ax)
#     rz1 = np.array((0, 0, 7.8))
#     arrow = np.sum(P1, axis=0) * 10
#     start_pos = rz1 - arrow
#     ax.arrow(start_pos[1], start_pos[2],
#              2 * arrow[1], 2 * arrow[2],
#              fc='k',
#              ec='k',
#              width=0.10,
#              head_width=0.25,
#              length_includes_head=False)
#     ax.plot(rz1[1], rz1[2], 'ro', ms=4)
#     ax.set_xlim(-15., 15.)
#     ax.set_ylim(-15., 15.)
#     ax.axis('off')
#     set_axis(ax, letter)

# plot analytically calculated potentials
X = params.x_points.reshape(180, 180)
Y = params.y_points.reshape(180, 180)
Z = params.z_points.reshape(180, 180)

rstride = 1
cstride = 1

#title_texts = ['Analytical', 'FEM', 'Error']
title_texts = ['', 'Analytical', 'FEM', '% error']
row_texts = ['Radial ', 'Tangential', 'Mixed']
#sphere_xlabel = [r'$\alpha = 0^\circ$', r'$\alpha = 90^\circ$', r'$\alpha = 45^\circ$']
sphere_xlabel = [r'', r'', r'']

def plot_phi_sphere(idx_val, letters, phi, error=None):
    for ii, letter in zip([0, 1, 2], letters):
        ax = plt.subplot(gs[ii, idx_val], projection='3d')
        if error is None:
            vmax = 10.  # abs_max_range[ii]
            vmin = -1 * vmax
            clrs = plt.cm.PRGn((phi[ii] - vmin) / (vmax - vmin))
        else:
            vmax = 0.3
            vmin = 0.0
            clrs = plt.cm.Greys((phi[ii] - vmin) / (vmax - vmin))
        ax.plot_surface(X, Y, Z,
                        rstride=rstride, cstride=cstride, linewidth=0,
                        facecolors=clrs,
                        antialiased=False)
        if ii == 0:
            ax.set_title(title_texts[ii], y=1.1)

        set_axis_3d(ax, letter)
        adjust_3d_axis(ax, ii)
        #ax.text2D(0.05, -0.15, row_texts[ii], transform=ax.transAxes, rotation='vertical')

def plot_phi(idx_val, letters, phi, error=None):
    for ii, letter in zip([0, 1, 2], letters):
        ax = plt.subplot(gs[ii, idx_val])
        if error is None:
            vmax = 10.  # abs_max_range[ii]
            vmin = -1 * vmax
            #clrs = plt.cm.PRGn((phi[ii] - vmin) / (vmax - vmin))
            cmap = cm.PRGn
            title_idx = 1
        

        else:
            vmax = 0.3
            vmin = 0.0
            cmap = cm.Greys
            #clrs = plt.cm.Greys((phi[ii] - vmin) / (vmax - vmin))
            title_idx = 2
        # ax.plot_surface(X, Y, Z,
        #                 rstride=rstride, cstride=cstride, linewidth=0,
        #                 facecolors=clrs,
        #                 antialiased=False)

        norm = cm.colors.Normalize(vmax=vmax, vmin=vmin, clip=False)
        im = plt.imshow(phi[ii], aspect='equal', norm=norm, cmap=cmap, interpolation='none')

        if error is None:
            CS = plt.contour(90.+params.phi_angle.reshape(180, 180),
                             params.theta.reshape(180, 180),
                             phi[ii], colors='k' , linewidths=0.25)
            plt.clabel(CS, fontsize=5, inline=1)

        if ii == 2:
            ax.set_xlabel(r'$ \phi $' + ' (degrees)')
        if idx_val == 1:
            ax.set_ylabel(r'$ \theta $' + ' (degrees)')

        if ii == 0:
            ax.set_title(title_texts[idx_val], y=1.1)

        plt.xticks(np.arange(0, 180, 30), np.arange(-90, 90, 30))
        plt.yticks(np.arange(0, 180, 30), np.arange(0, 180, 30)) 

        set_axis(ax, letter)
    return im



plot_phi_sphere(0, ['A', 'E', 'I'], ana_list)
im_phi = plot_phi(1, ['B', 'F', 'J'], ana_list)
im_phi_2 = plot_phi(2, ['C', 'G', 'K'], fem_list)
cax_phi = plt.subplot(gs[3, 0:3])
cbar_0 = plt.colorbar(im_phi, cax=cax_phi, extend='both', orientation='horizontal')
cbar_0.set_label(r'Potential ($\mathrm{\mu}$V)')

im_error = plot_phi(3, ['D', 'H', 'L'], error_list, error=True)
cax_error = plt.subplot(gs[3, 3])
cbar = plt.colorbar(im_error, cax=cax_error, orientation='horizontal',
                    ticks=[0.0, 0.1, 0.2, 0.3], extend='max')
cbar.ax.set_xticklabels(['0.0', '0.1', '0.2', '0.3'])
cbar.set_label(r'%')

plt.savefig(os.path.join(args.results,
                         'figure2_alt.jpeg'),
            dpi=500)
#plt.show()
