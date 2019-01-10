import numpy as np
import matplotlib.pyplot as plt
import os.path
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--directory', '-d',
                    default='results',
                    dest='results',
                    help='a path to the result directory')

args = parser.parse_args()

analy = np.load(os.path.join(args.results,
                             'Analytical_correct_all.npz'))
calc_ = np.load(os.path.join(args.results,
                             'CalcPotential4_correct_all.npz'))

for key_val in analy.keys():
    plt.close('all')
    plt.subplot(121)
    plt.imshow(analy[key_val].reshape(180, 180), cmap=plt.cm.PRGn)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(calc_[key_val], cmap=plt.cm.PRGn)
    plt.colorbar()
    plt.savefig(os.path.join(args.results,
                             key_val + '.png'))
