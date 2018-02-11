from const import *
from calc import *
from NN import *
from scipy.io import loadmat

np.random.seed(0)
M = loadmat("mnist_all.mat")

X = genX(M)
Y = genY(M)
