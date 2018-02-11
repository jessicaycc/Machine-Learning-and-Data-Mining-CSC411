import numpy as np
from numpy import dot
from numpy import log
from numpy import exp
from numpy.linalg import norm

EPS = 1e-5
LRN_RATE = 0.1
MAX_ITER = 30000
IMG_SHAPE = (28, 28)
SET_RATIO = (100, 10, 10)

LABEL = np.identity(10)