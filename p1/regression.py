from const import *
from numpy import dot
from numpy.linalg import norm

def Jbin(X, w, y):
    return norm(dot(X, w)-y)**2 / (2*len(X))
def dJbin(X, w, y):
    return dot(X.T, dot(X, w)-y) / len(X)

def J(X, W, Y):
    return norm(dot(W.T, X)-Y)**2
def dJ(X, W, Y):
    return 2*dot(X, (dot(W.T, X)-Y).T)

def genMatrix(fileList):
    M = np.empty((0, VEC_SIZE), float)
    for filename in fileList:
        img = Image.open("processed/" + filename)
        x = np.array(img).flatten() / 255.
        M = np.vstack((M, x))
    return np.append(M, np.ones((len(M), 1)), axis=1)

def gradDescent(f, df, X, w0, Y, alpha=LRN_RATE,
                intr=5000, maxIter=MAX_ITER, out=True):
    def update(f, df, X, W, Y, i):
        print "Iter", i
        print "J(x) =", f(X, W, Y) 
        print "Gradient:", df(X, W, Y), "\n"
        return

    i = 0
    W = w0.copy()
    wPrev = w0 - 10*EPS
    while norm(W - wPrev) > EPS and i < maxIter:
        wPrev = W.copy()
        W -= alpha * df(X, W, Y)
        if i % intr == 0 and out:
            update(f, df, X, W, Y, i)
        i += 1
    if out:
        update(f, df, X, W, Y, i)
    return W

def finiteDiff(f, X, W, Y, p, q, h=DIV):
    E = np.zeros((len(W), len(W[0])))
    E[p][q] = h
    return (f(X, W+E, Y) - f(X, W, Y)) / h

def percentDiff(a, b):
    return 2 * abs(a-b) / float(a+b) * 100

def accuracy(validSet, W, Y):
    def classify(filename, W, y):
        img = Image.open("processed/" + filename)
        x = np.array(img).flatten() / 255.
        res = dot(W.T, np.append(x, 1))
        th = max(res) if isinstance(res, list) else TSH_HOLD
        res = (res >= th).astype(int)
        return norm(res-y) == 0

    N = len(validSet)
    M = len(validSet[0])
    correct = 0
    for i in xrange(N):
        for filename in validSet[i]:
            if classify(filename, W, Y[i]):
                correct += 1
    return correct / float(M*N)
