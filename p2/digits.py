import os
from calc import *
from scipy.io import loadmat

def save(obj, filename):
    if not os.path.exists('objects'):
        os.makedirs('objects')
    cPickle.dump(obj, open('objects/'+filename+'.pk', 'wb'))
    return
def load(filename):
    return cPickle.load(open('objects/'+filename+'.pk', 'rb'))

np.random.seed(0)
M = loadmat("mnist_all.mat")

X = genX(M)
Y = genY(SET_RATIO[0])
W = np.zeros((NUM_LABEL, NUM_FEAT))
b = np.zeros((NUM_LABEL, 1))

#P = forward(X, W, b)

#n = dC_weight(X, Y, P)
#print np.shape(n)
#m = finiteDiff_weight(X, Y, W, b, 5, 157)
#print relativeError(n[5][157], m)

#n = dC_bias(X, Y, P)
#print np.shape(n)
#m = finiteDiff_bias(X, Y, W, b, 1)
#print relativeError(n[1][0], m)

#W, b = gradDescent(X, Y, W, b)
#save(W, 'weights')
#save(b, 'bias')

W, b = gradDescent(X, Y, W, b)
W, b = gradDescentMoment(X, Y, W, b)
