import os
import time
import _pickle
import numpy as np
from numpy import dot
from numpy import log
from numpy import exp
from numpy.linalg import norm

if not os.path.exists('objects'):
    os.makedirs('objects')

def saveObj(obj, filename):
    _pickle.dump(obj, open('objects/'+filename+'.obj', 'wb'))
    return

def loadObj(filename):
    return _pickle.load(open('objects/'+filename+'.obj', 'rb'))

np.random.seed(0)

# For parts 1-7
NUM_FEAT = 784
NUM_LABEL = 10
IMG_SHAPE = (28, 28)

STEP = 100
EPS = 1e-5
GAMMA = 0.99
ALPHA = 0.025
MAX_ITER = 10000

TEST = 'test'
TRAIN = 'train'
LABEL = np.identity(10)

# For parts 8-10
DATA_NAME = 0
DATA_URL = 3
DATA_BBOX = 4
DATA_HASH = 5

act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
