import numpy as np
from numpy import dot
from numpy import log
from numpy import exp
from numpy.linalg import norm

NUM_FEAT = 784
NUM_LABEL = 10
IMG_SHAPE = (28, 28)
SET_RATIO = (100, 10, 10)

LABEL = np.identity(10)

EPS = 1e-4
LRN_RATE = 0.01
MAX_ITER = 30000
