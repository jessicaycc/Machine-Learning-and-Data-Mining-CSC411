from const import *

def C(Y, P):
    return -sum(Y*log(P))

def dC_weight(X, Y, P):
    return dot((P-Y).T, X)

def dC_bias(X, Y, P):
    i = np.ones(len(X))
    return dot((P-Y).T, i)

def finiteDiff_weight(X, W, b, Y, i, j):
    def f(A):
        return C(Y, forward(X, A, b))
    E = np.zeros(np.shape(W))
    E[i][j] = EPS
    return (f(W+E)-f(W)) / EPS

def finiteDiff_bias(x, w, b, y, i):
    def f(a):
        return C(Y, forward(X, W, a))
    e = np.zeros(np.shape(b))
    e[i] = EPS
    return (f(b+e)-f(b)) / EPS

def relativeError(a, b):
    a, b = abs(a), abs(b)
    return 2*abs(a-b) / float(a+b)

def gradDescent(X, Y, W0, b0):
    i = 0
    W = W0.copy()
    b = b0.copy()
    WPrev = W0 - 10*EPS
    bPrev = b0 - 10*EPS
    while norm(W-WPrev)>EPS or norm(b-bPrev)>EPS and i<MAX_ITER:
        WPrev = W.copy()
        bPrev = b.copy()
        P = forward(X, W, b)
        W -= LRN_RATE * dC_weight(X, Y, P)
        b -= LRN_RATE * dC_bias(X, Y, P)
        if i % 500 == 0:
            print "Iter", i
            print "C(Y, P)", C(Y, P), '\n'
        i += 1
    return W, b