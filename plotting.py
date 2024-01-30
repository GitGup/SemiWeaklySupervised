import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display, clear_output
from PIL import Image, ImageSequence
import os

import seaborn as sea
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline

sea.set(style="white")

#plot landscape dynamically
def loss_landscape_nofit(sigfrac, m1, m2, z):
    
    #create grid of points for the model to be evaluated at
    #current resolution is 23 * 23 = 529
    
    start = 0.5
    end = 6
    step = 0.125

    weight_list = np.arange(start, end + step, step)
    
    grid_axes = []
    for w1 in weight_list:
        for w2 in weight_list:
            grid_axes.append((w1, w2))
            
    w1_values, w2_values = zip(*grid_axes)

    loss_values = list(z[sigfrac, m1, m2])
    min_loss = min(loss_values)
    max_loss = max(loss_values)

    star1_coords = (m1, m2)  
    star2_coords = (m2, m1)

    normalized_loss = [(x - min_loss) / (max_loss - min_loss) for x in loss_values]
    
    h = plt.hist2d(w1_values, w2_values, bins=(23,23), cmap='viridis', weights=normalized_loss)
    plt.scatter(*star1_coords, c='red', marker='*', s=200, label='Star 1')
    plt.scatter(*star2_coords, c='blue', marker='*', s=200, label='Star 2')
    plt.colorbar(label='Loss (BCE)')

    plt.xlabel('m1')
    plt.ylabel('m2')
    plt.title('6 Features (m1 = {} | m2 = {}) sigfrac: {:.4f}'.format(m1, m2, sigfrac))
    return h

#Loss Landscape but 3D
def create_3D_loss_manifold(sigfrac, m1, m2):

    start = 0.5
    end = 6
    step = 0.25

    weight_list = np.arange(start, end + step, step)

    grid_axes = []
    for w1 in weight_list:
        for w2 in weight_list:
            grid_axes.append((w1, w2))

    w1_values, w2_values = zip(*grid_axes)

    loss_values = list(z_allm1m2_HD[sigfrac, m1, m2])

    x = w1_values
    y = w2_values
    z = loss_values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='.', alpha = 0.3)
    ax.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')

    ax.set_xlabel('W1')
    ax.set_ylabel('W2')
    ax.set_zlabel('Loss Label')
    ax.set_title("3D Loss Manifold m1: {} m2: {} sigfrac: {np.round(sigfrac, 5)}".format(m1, m2))

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.grid(False)
    
    ax.view_init(elev=30, azim=10)
    plt.show()
