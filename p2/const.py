import numpy as np
from numpy import dot
from numpy import log
from numpy import exp
from numpy.linalg import norm

TEST = "test"
TRAIN = "train"

NUM_FEAT = 784
NUM_LABEL = 10
IMG_SHAPE = (28, 28)

MAX_TEST_SIZE = 892
MAX_TRAIN_SIZE = 5421

EPS = 1e-5
GAMMA = 0.99
ALPHA = 1e-9
#ALPHA = 0.025
ALPHA_M = 0.001
MAX_ITER = 10000

LABEL = np.identity(10)
