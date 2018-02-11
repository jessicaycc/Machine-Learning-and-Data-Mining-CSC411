from calc import *
from scipy.io import loadmat

np.random.seed(0)
M = loadmat("mnist_all.mat")

X = genX(M)
Y = genY(SET_RATIO[0])
W = np.zeros((NUM_LABEL, NUM_FEAT))
b = np.zeros((NUM_LABEL, 1))

#P = forward(X, W, b)

#n = dC_weight(X, Y, P)
#m = finiteDiff_weight(X, Y, W, b, 5, 157)
#print relativeError(n[5][157], m)

#n = dC_bias(X, Y, P)
#m = finiteDiff_bias(X, Y, W, b, 1)
#print relativeError(n[1][0], m)

W, b = gradDescent(X, Y, W, b)