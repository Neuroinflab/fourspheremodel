from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec, ticker
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import os
import parameters as params
from matplotlib import colors

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d',
                    default='results',
                    dest='results',
                    help='a path to the result directory')

args = parser.parse_args()

print('load data')
# import EEG on sphere computed analytically and numerically
# these are computed with current input = 1 microA
# dipole length = .1 cm (dipole moment units = 10^-9 Am)
# and give potentials with unit microV
# in order to get results in normal EEG magnitude: 10-100 micro V
# we give current input = 100 micro A
k = 100  # scaling factor --> give results in ~10 micro V
ana_rad = np.load(os.path.join(args.results, "Analytical_rad.npz"))['phi_20']*k
ana_tan = np.load(os.path.join(args.results, "Analytical_tan.npz"))['phi_20']*k
ana_mix = np.load(os.path.join(args.results, "Analytical_mix.npz"))['phi_20']*k

num_rad = np.load(os.path.join(args.results, "Numerical_rad.npz"))['fem_20']*k
num_tan = np.load(os.path.join(args.results, "Numerical_tan.npz"))['fem_20']*k
num_mix = np.load(os.path.join(args.results, "Numerical_mix.npz"))['fem_20']*k

fem_list = [num_rad, num_tan, num_mix]
ana_list = [ana_rad, ana_tan, ana_mix]

fem_list = [ii.reshape(180, 180) for ii in fem_list]
ana_list = [ii.reshape(180, 180) for ii in ana_list]
error_list = [np.abs(ii - jj) for ii, jj in zip(fem_list, ana_list)]

# RE_list = [np.abs(ii)/np.abs(jj) for ii, jj in zip(error_list, ana_list)]
# A = np.array(RE_list).flatten()
# for idx, i in enumerate(A):
#     if i > 1.:
#         A[idx] = 1.
# RE_list = A.reshape(3,180,180)

# abs_max_val = [np.max(np.abs(ii)) for ii in ana_list]
# abs_max_range = [0.3 * ii for ii in abs_max_val]
mean_abs_eeg = np.average(np.abs([ana_tan, ana_rad, ana_mix])) # average normalization
max_abs_eeg = np.max(np.abs([ana_tan, ana_rad, ana_mix])) # max normalization
# scaled_error_list0 = [error_list[i]/mean_abs_eeg for i in range(3)]  # average normalization
scaled_error_list = [error_list[i]/max_abs_eeg for i in range(3)] # max normalization
# scaled_error_list = [error_list[i]/max_1promille for i in range(3)]
# print 'Error max for rad, tan, mix: ', error_max
# print 'Abs max for rad, tan, min', [np.max(np.abs(ii)) for ii in ana_list]


I = 1.0*k
dipole_list = params.dipole_list
P1s = []
name_list = []
radii = [params.brain_rad, params.csftop_rad,
         params.skull_rad, params.scalp_rad]

for dipole in dipole_list:
    src = np.array(dipole["src_pos"])
    snk = np.array(dipole["snk_pos"])
    # P1s.append(np.array((src * I, -1 * snk * I)))
    P1s.append(np.array([src*I - snk*I]))
    name_list.append(dipole["name"])
rz1 = np.array([0, 0, 7.8])
# create array with 2D electrode positions for plotting geometry
print('creating positions for plotting 2d geometry')
Radii = [79000., 80000., 85000., 90000.]
R = 90000.
ys = np.linspace(-R, R, 1001)
zs = np.linspace(-R, R, 1001)
elec_pos = []
for zpos in zs:
    for ypos in ys:
        elec_pos.append([0., ypos, zpos])
elec_pos = np.array(elec_pos)

# mark subdomains for plotting geometry
subdomain_markers = []
for pos in elec_pos:
    rad = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    # print r
    if rad < Radii[0]-500:  #  or (r-79000.) < 1e-16:
        subdomain_markers.append(0.1)
    elif rad < Radii[1]: #  or (r-80000.) < 1e-16:
        subdomain_markers.append(0.2)
    elif rad < Radii[2]: #  or (r-90000.) < 1e-16:
        subdomain_markers.append(0.3)
    elif rad < Radii[3]:  # or (r-100000.) < 1e-16:
        subdomain_markers.append(0.4)
    else:
        subdomain_markers.append(np.nan)
subd_markers = np.array(subdomain_markers).reshape(len(ys), len(zs))


# create color lists for plotting of potentials and error
print('creating color lists')
vmax = 10
vmin = -vmax
clr = lambda phi: plt.cm.PRGn((phi - vmin) / (vmax - vmin))
vmax_error = .03
vmax_error_glob_max = 0.003
vmin_error = 0.
# clr_error = lambda error: plt.cm.Greys((error - vmin_error) / (vmax_error - vmin_error))
clr_error = lambda error: plt.cm.Greys((error - vmin_error) / (vmax_error_glob_max - vmin_error))

colors_4s_list = [clr(ana_list[idx]) for idx in range(3)]
colors_fem_list = [clr(fem_list[idx]) for idx in range(3)]
# colors_error_list = [clr_error(error_list[idx]) for idx in range(3)]
colors_scaled_error_list = [clr_error(scaled_error_list[idx]) for idx in range(3)]
# print 'max_RE', np.max(RE_list)
# colors_RE_list = [clr_RE(RE_list[idx]) for idx in range(3)]
print('colors_4s created, ready for plotting')

#############################################################################
################################# plotting ##################################
#############################################################################
plt.close('all')
fig = plt.figure()

ax1 = plt.subplot2grid((3,4),(0,0))
ax2 = plt.subplot2grid((3,4),(0,1), projection='3d')
ax3 = plt.subplot2grid((3,4),(0,2), projection='3d')
ax4 = plt.subplot2grid((3,4),(0,3), projection='3d')
ax5 = plt.subplot2grid((3,4),(1,0))
ax6 = plt.subplot2grid((3,4),(1,1), projection='3d')
ax7 = plt.subplot2grid((3,4),(1,2), projection='3d')
ax8 = plt.subplot2grid((3,4),(1,3), projection='3d')
ax9 = plt.subplot2grid((3,4),(2,0))
ax10 = plt.subplot2grid((3,4),(2,1), projection='3d')
ax11 = plt.subplot2grid((3,4),(2,2), projection='3d')
ax12 = plt.subplot2grid((3,4),(2,3), projection='3d')
ax2.set_title('Analytical', y = 1.1)
ax3.set_title('FEM', y = 1.1)
# ax4.set_title('Relative error', y = 1.1)
ax4.set_title('Error scaled', y = 1.1)

# plot geometry
# print 'plotting geometry'
for ax, P1 in zip([ax1, ax5, ax9], P1s):
    im = ax.imshow(subd_markers.reshape(len(ys), len(zs)).T,
                    origin='lower', interpolation='nearest', extent = [-90000., 90000., -90000., 90000.])
    cmap = colors.ListedColormap(['#ffffff', '#83caff', '#999999', '#ffcc99'])
    im.set_cmap(cmap)
    ax.add_patch(plt.Circle((0, 0), radius = Radii[0]-500, color = 'gray', fill=False, lw = .3))
    ax.add_patch(plt.Circle((0, 0), radius = Radii[1], color = 'gray', fill=False, lw = .3))
    ax.add_patch(plt.Circle((0, 0), radius = Radii[2], color = 'gray', fill=False, lw = .3))
    ax.add_patch(plt.Circle((0,0), radius = Radii[3], color = 'gray', fill=False, lw = .5))
    ax.arrow(-100000, -100000, 0, 18000, head_width=4000, head_length=4000, fc='k', ec='k')
    ax.arrow(-100000, -100000, 18000,0, head_width=4000, head_length=4000, fc='k', ec='k')
    ax.arrow(-100000, -100000, -7800, -7800, head_width=4000, head_length=3300, fc='k', ec='k')
    ax.text(-102800, -70000, 'z', size = 6.)
    ax.text(-70000, -103800, 'y', size = 6.)
    ax.text(-115000, -120000, 'x', size = 6.)
    ax.set_xlim(-120000., 120000.)
    ax.set_ylim(-120000., 120000.)

    rz = rz1*1e4
    arrow = np.sum(P1, axis = 0)*1800
    start_pos = rz -np.array([0,0,500])- arrow
    print('start-pos',  start_pos)
    print('arrow', arrow)
    print('P1', P1)
    ax.arrow(start_pos[1], start_pos[2],
              2*arrow[1], 2*arrow[2],
              fc='k',
              ec='k',
              width = 170,
              head_width = 5000.,
              length_includes_head=False,
              )
    ax.plot(rz[1], rz[2]-100, 'ro', ms=4)
    ax.axis('off')
# plot analytically calculated potentials
print('reshaping XYZ')
X = params.x_points.reshape(180, 180)
Y = params.y_points.reshape(180, 180)
Z = params.z_points.reshape(180, 180)

for ax, clrs in zip([ax2, ax6, ax10], colors_4s_list):
    surf1 = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=clrs,
                             linewidth=0, antialiased=False)
print('surf1 OK')
# plot FEM-computed potentials
for ax, clrs in zip([ax3, ax7, ax11], colors_fem_list):
    surf2 = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=clrs,
                             linewidth=0, antialiased=False)
print('surf2 OK')
# plot errors
# for ax, clrs in zip([ax4, ax8, ax12], colors_error_list):
for ax, clrs in zip([ax4, ax8, ax12], colors_scaled_error_list):
# for ax, clrs in zip([ax4, ax8, ax12], colors_RE_list):
    surf3 = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = clrs,#cmap=plt.cm.Blues,
                   linewidth=0, antialiased=False)  #, norm=colors.LogNorm(vmin=1e-7, vmax=1e-2))
print('surf3 OK')
for ax in [ax2, ax3, ax4, ax6, ax7, ax8, ax10, ax11, ax12]:
    ax.set_aspect('equal')
    ax.axis('off')
    # ax.auto_scale_xyz([-60000., 60000.], [-60000., 60000.], [-60000., 60000.])
    ax.set_xlim3d(-6.5000, 6.5000)
    ax.set_ylim3d(-6.5000, 6.5000)
    ax.set_zlim3d(0.25-6.5000, 0.25+6.5000)
    ax.view_init(10, 0)
print('axes ok')


# colorbars
cax1 = fig.add_axes([0.31, 0.07, 0.38, 0.01])
m = plt.cm.ScalarMappable(cmap=plt.cm.PRGn)

ticks = np.linspace(-vmax,vmax, 9)

m.set_array(ticks)
cbar1 = fig.colorbar(m, cax=cax1, format='%3.3f', extend = 'both', orientation='horizontal')
cbar1.outline.set_visible(False)

cbar1.set_ticks(ticks)
cax1.set_xticklabels([str(i) for i in ticks], rotation=40.)
cbar1.set_label(r'Potential ($\mathrm{\mu}$V)', labelpad=1.)

cax2 = fig.add_axes([0.79, 0.07, 0.15, 0.01])
m = plt.cm.ScalarMappable(cmap=plt.cm.Greys)
ticks2 = np.linspace(vmin_error, vmax_error_glob_max, 4) # global normalization

m.set_array(ticks2)
cbar2 = fig.colorbar(m, cax=cax2, format='%3.6f', extend='max', orientation='horizontal')
cbar2.outline.set_visible(False)

cbar2.set_ticks(ticks2)
cax2.set_xticklabels([str(i*100) for i in ticks2], rotation=40.)

cbar2.set_label(r'%', labelpad=9.1)

fig.subplots_adjust(wspace = -.1, hspace=-.1, right = 0.98, left = 0.01)

# label figures
fig.text(0.03, .87, 'A',
        horizontalalignment='center',
        verticalalignment='center',
        fontweight='demibold',
        fontsize=12)
fig.text(0.27, .87, 'B',
        horizontalalignment='center',
        verticalalignment='center',
        fontweight='demibold',
        fontsize=12)
fig.text(0.51, .87, 'C',
        horizontalalignment='center',
        verticalalignment='center',
        fontweight='demibold',
        fontsize=12)
fig.text(0.75, .87, 'D',
        horizontalalignment='center',
        verticalalignment='center',
        fontweight='demibold',
        fontsize=12)
fig.text(0.03, .595, 'E',
        horizontalalignment='center',
        verticalalignment='center',
        fontweight='demibold',
        fontsize=12)
fig.text(0.27, .595, 'F',
        horizontalalignment='center',
        verticalalignment='center',
        fontweight='demibold',
        fontsize=12)
fig.text(0.51, .595, 'G',
        horizontalalignment='center',
        verticalalignment='center',
        fontweight='demibold',
        fontsize=12)
fig.text(0.75, .595, 'H',
        horizontalalignment='center',
        verticalalignment='center',
        fontweight='demibold',
        fontsize=12)
fig.text(0.03, .34, 'I',
        horizontalalignment='center',
        verticalalignment='center',
        fontweight='demibold',
        fontsize=12)
fig.text(0.27, .34, 'J',
        horizontalalignment='center',
        verticalalignment='center',
        fontweight='demibold',
        fontsize=12)
fig.text(0.51, .34, 'K',
        horizontalalignment='center',
        verticalalignment='center',
        fontweight='demibold',
        fontsize=12)
fig.text(0.75, .34, 'L',
        horizontalalignment='center',
        verticalalignment='center',
        fontweight='demibold',
        fontsize=12)


fig.set_size_inches(6.5, 6.)
# plt.show()
print('saving figure')
# fig.tight_layout()
# plt.savefig(os.path.join(args.results,
#                          'figure2_w_scaled_glob_avg_strength_error.png'),
#             dpi=300., bbox_inches='tight')
plt.savefig(os.path.join(args.results,
                         'figure2_w_scaled_glob_max5.png'),
            dpi=600., bbox_inches='tight')
# plt.savefig(os.path.join(args.results,
#                          'figure2_w_RE.png'),
#             dpi=300., bbox_inches='tight')
