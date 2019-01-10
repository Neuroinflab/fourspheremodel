import os
import numpy as np
import matplotlib.pyplot as plt
from plotting_convention import mark_subplots, simplify_axes
import parameters as params

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d',
                    default='results',
                    dest='results',
                    help='a path to the result directory')
parser.add_argument('--sri98-no-bn1',
                    action='store_const',
                    const='Sri98_no_bn1',
                    default='Sri98',
                    dest='sri98',
                    help='a path to the result directory')

args = parser.parse_args()

# Homogeneous infinite medium
k = 100. # scaling factor --> give results in ~10 micro V
I = 1.*k
d = .1
phi_0 = I*d / (4*np.pi*params.sigma_brain*(params.scalp_rad**2))

theta = params.theta.reshape(180, 180)[:, 0][0:90]

phi_20_fit = (34.1*np.exp(-0.15*theta) + 1.29 - (0.123*theta) + 0.00164*(theta**2))*phi_0
phi_40_fit = (27.4*np.exp(-0.10*theta) - 5.49 + (0.203*theta) - 0.00234*(theta**2))*phi_0
phi_80_fit = (13.4*np.exp(-0.10*theta) - 0.155 - (0.0135*theta))*phi_0

nunsri06 = np.load(os.path.join(args.results, 'Analytical_NunSri06_rad.npz'))
sri98 = np.load(os.path.join(args.results,
                             'Analytical_{.sri98}_rad.npz'.format(args)))
analytical = np.load(os.path.join(args.results, 'Analytical_rad.npz'))
numerical = np.load(os.path.join(args.results, 'Numerical_rad.npz'))

phi_20 = nunsri06['phi_20'].reshape(180, 180)[:, 0][0:90]*k
phi_20_98 = sri98['phi_20'].reshape(180, 180)[:, 0][0:90]*k
phi_20_c = analytical['phi_20'].reshape(180, 180)[:, 0][0:90]*k
num_20 = numerical['fem_20'].reshape(180, 180)[:, 0][0:90]*k

phi_40 = nunsri06['phi_40'].reshape(180, 180)[:, 0][0:90]*k
phi_40_98 = sri98['phi_40'].reshape(180, 180)[:, 0][0:90]*k
phi_40_c = analytical['phi_40'].reshape(180, 180)[:, 0][0:90]*k
num_40 = numerical['fem_40'].reshape(180, 180)[:, 0][0:90]*k

phi_80 = nunsri06['phi_80'].reshape(180, 180)[:, 0][0:90]*k
phi_80_98 = sri98['phi_80'].reshape(180, 180)[:, 0][0:90]*k
phi_80_c = analytical['phi_80'].reshape(180, 180)[:, 0][0:90]*k
num_80 = numerical['fem_80'].reshape(180, 180)[:, 0][0:90]*k

fig = plt.figure(figsize=(12, 4))
fig.subplots_adjust(wspace = 0.28, bottom=0.15, left=0.08, right=0.99)

# ylim = [-0.050000000000000003, 0.25000000000000006]
# xlim = [0., 80.00000000000000003]

ylim = [-4.99999999999, 110.000000000000000001]
xlim = [0., 80.00000000000000003]

ax = plt.subplot(131, ylim=ylim, xlim=xlim)
ax.set_title(r'$\sigma_{\mathrm{skull}} = \sigma_{\mathrm{brain}} / 20$', fontsize=17)
# ax.set_ylabel('Potential (mV)', fontsize=14)
ax.set_ylabel(r'Potential ($\mathrm{\mu V}$)', fontsize=14)
plt.plot(theta, phi_20, 'k', lw=2)
plt.plot(theta, phi_20_98, 'g', lw=2)
plt.plot(theta[:50], phi_20_fit[:50], 'm+', lw=2)
plt.plot(theta, phi_20_c, 'b', lw=2)
plt.plot(theta, num_20, 'r.', lw=2)
ax.set_xticks(ax.get_xticks()[::2])
ax.tick_params(labelsize=15.)
ax.set_yticks(np.linspace(0, 110, 12))
ax.set_yticklabels(['0', '', '20', '', '40', '', '60', '', '80', '',
                    '100', ''])

ax = plt.subplot(132, ylim=ylim, xlim=xlim)
ax.set_xlabel('Polar angle (degrees)', fontsize=14)
ax.set_title(r'$\sigma_{\mathrm{skull}} = \sigma_{\mathrm{brain}} / 40$', fontsize=17)
plt.plot(theta, phi_40, 'k', lw=2)
plt.plot(theta, phi_40_98, 'g', lw=2)
plt.plot(theta[:50], phi_40_fit[:50], 'm+', lw=2)
plt.plot(theta, phi_40_c, 'b', lw=2)
plt.plot(theta, num_40, 'r.', lw=2)
ax.set_xticks(ax.get_xticks()[::2])
ax.tick_params(labelsize=15.)
ax.set_yticks(np.linspace(0, 110, 12))
ax.set_yticklabels(['0', '', '20', '', '40', '', '60', '', '80', '',
                    '100', ''])

ax = plt.subplot(133, ylim=ylim, xlim=xlim)
ax.set_title(r'$\sigma_{\mathrm{skull}} = \sigma_{\mathrm{brain}} / 80$', fontsize=17)
l1, = plt.plot(theta, phi_80_98, 'g', lw=2)
l2, = plt.plot(theta, phi_80, 'k', lw=2)
l3, = plt.plot(theta[:50], phi_80_fit[:50], 'm+', lw=2)
l4, = plt.plot(theta, phi_80_c, 'b', lw=2)
l5, = plt.plot(theta, num_80, 'r.', lw=2)
ax.set_xticks(ax.get_xticks()[::2])
ax.tick_params(labelsize=15.)
ax.set_yticks(np.linspace(0, 110, 12))
ax.set_yticklabels(['0', '', '20', '', '40', '', '60', '', '80', '',
                    '100', ''])

lines = [l1, l2, l3, l4, l5]
line_names = ['Srinivasan (1998)', 'Nunez & Srinivasan (2006)',
              'Fit - Nunez & Srinivasan (2006) ', 'Present results - analytical', 'Present results - FEM']
fig.legend(lines, line_names, frameon=False, ncol=1, bbox_to_anchor=[1.01, 0.85], fontsize=12)

simplify_axes(fig.axes)
mark_subplots(fig.axes, ypos=1.07, xpos=-0.)
# plt.savefig(os.path.join(args.results, 'Potentials_scaled.png'), dpi=150)
# plt.savefig(os.path.join(args.results, 'figure3.eps'), dpi=150)
# plt.show()
