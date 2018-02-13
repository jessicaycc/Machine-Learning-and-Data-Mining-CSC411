import numpy as np
from numpy import dot
from numpy import log
from numpy import exp
from numpy.linalg import norm

NUM_FEAT = 784
NUM_LABEL = 10
IMG_SHAPE = (28, 28)

MAX_TEST_SIZE = 850
MAX_TRAIN_SIZE = 5000

STEP = 100
EPS = 1e-5
GAMMA = 0.99
ALPHA = 1e-9
MAX_ITER = 10000

LABEL = np.identity(10)
TEST = ('test0', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9')
TRAIN = ('train0', 'train1', 'train2', 'train3', 'train4', 'train5', 'train6', 'train7', 'train8', 'train9')
