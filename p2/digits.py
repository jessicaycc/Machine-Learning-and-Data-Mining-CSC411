from const import *
from calc import *
from NN import *
from scipy.io import loadmat

np.random.seed(0)
M = loadmat("mnist_all.mat")

X = genX(M)
Y = genY(SET_RATIO[0])

#x = np.array([ M["train0"][150].reshape(IMG_SHAPE).flatten() / 255. ]).T
#y = np.array([ [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] ]).T