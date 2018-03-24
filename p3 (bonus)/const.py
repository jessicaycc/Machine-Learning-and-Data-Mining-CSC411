import os
import time
import torch
import _pickle
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

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
TL = torch.LongTensor
PAD_WORD = '<>'
NUM_FAKE = 5000
NUM_REAL = 4893
MAX_HL_LEN = 10
SET_RATIO = (0.70, 0.15, 0.15)
