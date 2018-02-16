import numpy as np
from numpy import dot
from numpy import log
from numpy import exp
from numpy.linalg import norm

NUM_FEAT = 784
NUM_LABEL = 10
IMG_SHAPE = (28, 28)

STEP = 100
EPS = 1e-5
GAMMA = 0.99
ALPHA = 0.025
MAX_ITER = 10000

TEST = "test"
TRAIN = "train"
LABEL = np.identity(10)
