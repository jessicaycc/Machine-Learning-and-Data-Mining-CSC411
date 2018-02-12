import numpy as np
from numpy import dot
from numpy import log
from numpy import exp
from numpy.linalg import norm

TEST = 'test'
TRAIN = 'train'

NUM_FEAT = 784
NUM_LABEL = 10
IMG_SHAPE = (28, 28)

LABEL = np.identity(10)

EPS = 1e-5
ALPHA = 0.001
GAMMA = 0.999
MAX_ITER = 30000

