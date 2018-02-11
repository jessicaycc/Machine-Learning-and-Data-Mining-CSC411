from const import *
from pylab import *
from numpy import dot
from numpy import log
from scipy.io import loadmat

def softmax(o):
    return exp(o)/sum(exp(o))

def forward(x, w, b):
    return softmax(dot(w.T, x) + b)

def C(y, p):
    return -sum(y*log(p))

def dC_weight(x, y, p):
    return dot(x, (p-y).T)

def dC_bias(y, p):
    return p-y

def finiteDiff_weight(x, w, b, y, i, j):
    def f(n):
        return C(y, forward(x, n, b))
    e = np.zeros(np.shape(w))
    e[i][j] = EPS
    return (f(w+e)-f(w)) / EPS

def finiteDiff_bias(x, w, b, y, i):
    def f(n):
        return C(y, forward(x, w, n))
    e = np.zeros(np.shape(b))
    e[i] = EPS
    return (f(b+e)-f(b)) / EPS

def relativeError(a, b):
    a, b = abs(a), abs(b)
    return 2*abs(a-b) / float(a+b)


np.random.seed(0)
M = loadmat("mnist_all.mat")

x = np.array([ M["train0"][150].reshape(IMG_SHAPE).flatten() / 255. ]).T
y = np.array([ [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] ]).T
w = np.random.rand(len(x), len(y))
b = np.random.rand(len(y), 1)

n = dC_weight(x, y, forward(x, w, b))
m = finiteDiff_weight(x, w, b, y, 153, 5)
print relativeError(n[153][5], m)

n = dC_bias(y, forward(x, w, b))
m = finiteDiff_bias(x, w, b, y, 1)
print relativeError(n[1][0], m)

#Display the 150-th "5" digit from the training set
#imshow(M["train0"][150].reshape(IMG_SHAPE), cmap=cm.gray)
#show()
