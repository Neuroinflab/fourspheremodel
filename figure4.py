# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from plotting_convention import mark_subplots, simplify_axes
import parameters as params
import sys
#from importlib import reload
#reload(sys)
#sys.setdefaultencoding('UTF-8')

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

# Homogeneous sphere, Nunez 2006, Eq. 6.7
k = 100 #to get answer in 10-100 µV
I = 1.*k
d = .1



f = params.dipole_loc / params.scalp_rad
K = 1.+ (f**2) - (2*f*np.cos(params.theta_r))
frst_trm = (2.*(np.cos(params.theta_r) - f)) / (K**1.5)
scnd_trm = (1./f)*((1/(K)**0.5) - 1.)
prod = frst_trm + scnd_trm
phi_sphere = I*d*prod / (4*np.pi*params.sigma_brain*(params.scalp_rad**2))

nunsri06 = np.load(os.path.join(args.results, 'Analytical_NunSri06_rad.npz'))
sri98 = np.load(os.path.join(args.results,
                             'Analytical_{.sri98}_rad.npz'.format(args)))
analytical = np.load(os.path.join(args.results, 'Analytical_rad.npz'))

phi_nunsri06 = nunsri06['phi_lim'].reshape(180, 180)[:, 0][0:90] * k
phi_sri98 = sri98['phi_lim'].reshape(180, 180)[:, 0][0:90] * k
phi_correct = analytical['phi_lim'].reshape(180, 180)[:, 0][0:90] * k

phi_sphere = phi_sphere.reshape(180, 180)[:, 0][0:90]
theta = params.theta.reshape(180, 180)[:, 0][0:90]

fig = plt.figure(figsize=[5, 4])
fig.subplots_adjust(left=0.2, bottom=0.14)
plt.subplot(111)
plt.xlim([0, 80])
plt.ylim([-15, 500.])

# plt.ylim([-0.5, 1.+1e-10]) # , xlim=[0, 50], ylim=[-0.2, 5])
# plt.plot(theta, phi_20, 'k', label='Nunez 20')

plt.plot(theta, phi_sphere, 'r', label='Homogeneous sphere')
plt.plot(theta, phi_nunsri06, 'k+', label='Nunez & Srinivasan (2006)')
plt.plot(theta, phi_sri98, 'g*', label='Srinivasan (1998)')
plt.plot(theta, phi_correct, 'b.', label='Present results - analytical')

plt.xlabel('Polar angle (degrees)', fontsize=14)
plt.ylabel(r'Potential ($\mathrm{\mu V}$)', fontsize=14.)
plt.tick_params(labelsize=15.)
plt.title('$\sigma_{\mathrm{skull}} = \sigma_{\mathrm{brain}} = \sigma_{\mathrm{csf}} = \sigma_{\mathrm{scalp}}$', fontsize=19)
plt.legend(frameon=False, bbox_to_anchor=(1, 0.9), fontsize=11)
simplify_axes(fig.axes)

# plt.savefig(os.path.join(args.results, 'figure4_scaled.png'), dpi=150)
plt.savefig(os.path.join(args.results, 'figure4.eps'), dpi=150)
# plt.show()
