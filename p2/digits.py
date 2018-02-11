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

def gradDescent(f, df, x, w0, y, p, alpha=LRN_RATE, intr=5000, maxIter=MAX_ITER, out=True):
    def update(f, df, x, y, p, i):
        print "Iter", i
        print "J(x) =", f(y, p) 
        print "Gradient:", df(x, y, p), "\n"
        return

    i = 0
    W = w0.copy()
    wPrev = w0 - 10*EPS
    while norm(W - wPrev) > EPS and i < maxIter:
        wPrev = W.copy()
        W -= alpha * df(x, y, p)
        if i % intr == 0 and out:
            update(f, df, x, y, p, i)
        i += 1
    if out:
        update(f, df, x, y, p, i)
    return W

np.random.seed(0)
M = loadmat("mnist_all.mat")

x = np.array([ M["train0"][150].reshape(IMG_SHAPE).flatten() / 255. ]).T
y = np.array([ [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] ]).T
w = np.random.rand(len(x), len(y))
b = np.random.rand(len(y), 1)

n = dC_weight(x, y, forward(x, w, b))
m = finiteDiff_weight(x, w, b, y, 153, 5)
print relativeError(n[153][5], m)
w0 = np.zeros((len(x), len(y)))
print (gradDescent(C, dC_weight, x, w0, y, forward(x ,w0, b)))


n = dC_bias(y, forward(x, w, b))
m = finiteDiff_bias(x, w, b, y, 1)
print relativeError(n[1][0], m)

#Display the 150-th "5" digit from the training set
#imshow(M["train0"][150].reshape(IMG_SHAPE), cmap=cm.gray)
#show()
