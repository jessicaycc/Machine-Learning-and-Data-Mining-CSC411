import os
import time
import torch
import _pickle
import numpy as np
import torch.nn as nn
import torch.utils.data

from numpy import dot
from numpy import log
from numpy import exp
from numpy.linalg import norm
from torch.autograd import Variable
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

np.random.seed(0)
torch.manual_seed(0)

if not os.path.exists('plots'):
    os.makedirs('plots')

if not os.path.exists('objects'):
    os.makedirs('objects')

def saveObj(obj, filename):
    _pickle.dump(obj, open('objects/'+filename+'.pkl', 'wb'))
    return

def loadObj(filename):
    return _pickle.load(open('objects/'+filename+'.pkl', 'rb'))

TF = torch.FloatTensor
NUM_FAKE = 1298
NUM_REAL = 1968
SET_RATIO = (0.70, 0.15, 0.15)
