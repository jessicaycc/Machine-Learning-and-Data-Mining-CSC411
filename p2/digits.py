from const import *
from numpy import dot
from numpy import log

import os
import time
import urllib
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cbook as cbook
from pylab import *
from numpy import random
from scipy.io import loadmat
from scipy.misc import imread
from scipy.misc import imresize
from scipy.ndimage import filters

def softmax(o):
    return exp(o) / tile(sum(exp(o),0), (len(o),1))

def forward(x, w, b):
    temp = softmax(dot(w.T, x) + b)
    return temp

def C(y, p):
    return -sum(y*log(p))

def dC_weight(x, y, p):
    return dot(p-y, x.T)

def dC_bias(b, y, p):
    return dot(p-y, np.ones(shape(b)).T)

def finiteDiff_weight(x, w, b, y, i, j):
    def f(n):
        return C(y, forward(x, n, b))
    e = np.zeros(np.shape(w))
    e[i][j] = h
    return (f(w+e)-f(w)) / EPS

def finiteDiff_bias(x, w, b, y, i):
    def f(n):
        return C(y, forward(x, w, n))
    e = np.zeros(np.shape(b))
    e[i] = h
    return (f(b+e)-f(b)) / EPS

def relativeError(a, b):
    a, b = abs(a), abs(b)
    return 2*abs(a-b) / float(a+b)


M = loadmat("mnist_all.mat")

x = M["train0"][150].reshape(IMG_SHAPE).flatten() / 255.
y = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
w = np.random.rand(len(x), len(y))
b = np.random.rand(len(y))

n = dC_weight(x, y, forward(x, w, b))
m = finiteDiff_weight(x, w, b, y, 4, 10)

print relativeError(n[4][10], m)

#Display the 150-th "5" digit from the training set
#imshow(M["train0"][150].reshape(IMG_SHAPE), cmap=cm.gray)
#show()
