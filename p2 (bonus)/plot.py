import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.pyplot import clabel

if not os.path.exists('plots'):
    os.makedirs('plots')

def convert(A, range_old, range_new):
    len_old = range_old[1]-range_old[0]
    len_new = range_new[1]-range_new[0]
    old_min = np.ones(A.shape)*range_old[0]
    new_min = np.ones(A.shape)*range_new[0]
    return (A - old_min)*len_new/len_old + new_min

def visWeights(W):
    for i, im in enumerate(W):
        name = 'weight_'+str(i)
        im = convert(im.T, (-1,1), (0,255))
        R = im[:,:,0]
        G = im[:,:,1]
        B = im[:,:,2]
        im = R + G + B
        plt.imshow(im, cmap=cm.coolwarm)
        plt.savefig('plots/'+name+'.png', bbox_inches='tight')
        plt.show()
    return
