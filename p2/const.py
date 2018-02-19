import numpy as np
from numpy import dot
from numpy import log
from numpy import exp
from numpy.linalg import norm

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

TEST = "test"
TRAIN = "train"
LABEL = np.identity(10)

# For parts 8-10
LORR = KRIS = 0
PERI = FRAN = 1
ANGI = AMER = 2
ALEC = DANI = 3
BILL = GERA = 4
STEV = MICH = 5

DATA_NAME = 0
DATA_URL = 3
DATA_BBOX = 4
DATA_HASH = 5
DATA_SIZE = (32, 32)
DATA_SET_RATIO = (60, 0, 20)

NUM_ACT = 6
VEC_SIZE = 3072
