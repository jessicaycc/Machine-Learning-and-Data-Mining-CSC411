import os
import time
import _pickle
import numpy as np
from numpy import dot
from numpy import log
from numpy import exp
from numpy.linalg import norm

np.random.seed(0)

if not os.path.exists('objects'):
    os.makedirs('objects')

def saveObj(obj, filename):
    _pickle.dump(obj, open('objects/'+filename+'.pkl', 'wb'))
    return

def loadObj(filename):
    return _pickle.load(open('objects/'+filename+'.pkl', 'rb'))

DATA_SET_RATIO = (0.70, 0.15, 0.15)
