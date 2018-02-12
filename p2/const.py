import numpy as np
from numpy import dot
from numpy import log
from numpy import exp
from numpy.linalg import norm

TEST = "test"
<<<<<<< HEAD
TRAIN = 'train'
=======
TRAIN = "train"
>>>>>>> 9a91d7fb3e7894e674a4c56bf5ee88604f1c1772

NUM_FEAT = 784
NUM_LABEL = 10
IMG_SHAPE = (28, 28)

EPS = 1e-5
GAMMA = 0.99
ALPHA = 0.025
ALPHA_M = 0.001
MAX_ITER = 30000

LABEL = np.identity(10)
