# Basic imports that should come with all official python distributions
import importlib
import time

# Standard python imports that should come with anaconda
# Else, they can be easily pip installed
# pip install numpy pandas scipy matplotlib ipython ipywidgets
import numpy as np
import pandas as pd

import scipy.ndimage as ndimage
import scipy.spatial.distance as distance

import warnings
warnings.filterwarnings( "ignore")
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# A comfy way to import/export TIFF files as numpy.arrays
# pip install tifffile
import tifffile as tf


# Our implementation of the Euler Characteristic Transform

# See the README in the repository to install `demeter`

import demeter.euler as euler
import demeter.directions as dirs
import demeter.misc as misc

seed_file = '/home/samuel/brainMorpho/THE_SCANS/only_brain/256_female.tiff'

#If running from colab, uncomment the following line
#leaf_file = 'gdrive/My Drive/tda_workshop/example_data/seed_8_0_p7_d4_t120_o7_e1_g3.tif'

seed_img = tf.imread(seed_file)
seed_img[seed_img > 0] = 1

tic = time.perf_counter()
seed = euler.CubicalComplex(seed_img).complexify()
toc = time.perf_counter()

print("Complexify in {:.4f} seconds.\n\nCubical complex made of:".format(toc-tic))
seed.summary();

max_vox = misc.find_tip(seed.cells[0], 2,1,0)
seed_coords,_, _, _,_ = misc.rotateSVD(seed.cells[0], max_vox)

misc.plot_3Dprojections(seed_coords, 'Sample barley seed')

sphere_dirs = dirs.regular_directions(128)
title = 'Regular placement: {}'.format(len(sphere_dirs))
misc.plot_3Dprojections(sphere_dirs, title, markersize=12)


## Plot functions so that we do not clutter the next slides/cells

def plot_ecc(CComplex, filtration, T, title='title', ylim=(-30,30)):
    fig, ax = plt.subplots(2,3, figsize=(15,6), facecolor='snow')
    fig.suptitle('ECC for filter: {}'.format(title), fontsize=30)
    for j in range(2):
        for k in range(3):
            ax[j,k].plot((0,0),ylim, c='white')
            ax[j,k].set_ylabel('Euler characteristic', fontsize=12)
            ax[j,k].set_xlabel('Sublevel set', fontsize=12)
            ax[j,k].plot(CComplex.ECC(filtration, T[3*j+k]), 
                         lw=3, label = 'T = {}'.format(T[3*j+k]))
            ax[j,k].legend(fontsize=14)
            ax[j,k].set_facecolor('gainsboro')
    fig.tight_layout()

def plot_ecc_filtration(CComplex, filtration, T, TT=32, title='title', ydot=(-30,30), direction=None):
    bins = np.linspace(np.min(filtration), np.max(filtration), TT+1)
    indices = np.digitize(filtration, bins=bins, right=False)

    fig = plt.figure(constrained_layout=True, figsize=(20,7), facecolor='snow')
    gs = fig.add_gridspec(2,4)
    for j in range(2):
        for k in range(2):
            ax = fig.add_subplot(gs[j,k])
            ax.plot((0,0),ydot, c='white')
            ax.set_ylabel('Euler characteristic', fontsize=12)
            ax.set_xlabel('Sublevel set', fontsize=12)
            ax.plot(CComplex.ECC(filtration, T[2*j+k]), lw=5, label = 'T = {}'.format(T[2*j+k]))
            ax.legend(fontsize=20)
            ax.set_facecolor('gainsboro')
    ax = fig.add_subplot(gs[:,2:])
    scatter = ax.scatter(*CComplex.cells[0].T, 
                         s=0.2, c=indices, cmap='plasma', label='T = {}'.format(TT))
    ax.set_facecolor('whitesmoke')
    if direction is not None:
        start = 0.5*(np.max(CComplex.cells[0], axis=0) + np.min(CComplex.cells[0], axis=0))
        norm = 0.5*np.linalg.norm(start-np.max(leaf.cells[0], axis=0))
        #start += 0.5*norm*direction
        end = norm*direction
        ax.arrow(*start, *end, head_width=30, head_length=20, width=4, color='blue')
        
    ax.legend(fontsize=20, markerscale=30)
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.01);
    ax.axis('equal');
    cbar.ax.tick_params(labelsize=20)

    fig.suptitle('ECC for filter: {}'.format(title), fontsize=30);


T = [4,8,16,32,48,64]
i = 0
direction = sphere_dirs[i,:]
heights = np.sum(seed_coords*direction, axis=1)

plot_ecc(seed, heights, T, title=np.around(direction,2))



t = 32
tic = time.perf_counter()
ect = seed.ECT(sphere_dirs, T=t, verts=seed_coords)
toc = time.perf_counter()
print("Complex with {} vertices\n".format(len(seed_coords)))
print("ECT with {} directions in {:.4f} seconds.\n{:.4f}s per direction".format(len(sphere_dirs), toc-tic, (toc-tic)/len(sphere_dirs)))

plt.figure(figsize=(15,2))
plt.plot(ect); 
plt.title('ECT for {} directions and {} thresholds each'.format(len(sphere_dirs),t), fontsize=24);