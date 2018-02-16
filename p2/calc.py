from NN import *

def C(Y, P):
    return -np.sum(Y*log(P))

def dC_weight(X, Y, P):
    return dot((P-Y).T, X)

def dC_bias(X, Y, P):
    i = np.ones((len(X), 1))
    return dot((P-Y).T, i)

def finiteDiff_weight(X, Y, W, b, i, j):
    def f(A):
        return C(Y, forward(X, A, b))
    E = np.zeros(np.shape(W))
    E[i][j] = EPS
    return (f(W+E)-f(W)) / EPS

def finiteDiff_bias(X, Y, W, b, i):
    def f(a):
        return C(Y, forward(X, W, a))
    e = np.zeros(np.shape(b))
    e[i] = EPS
    return (f(b+e)-f(b)) / EPS

def relativeError(a, b):
    a, b = abs(a), abs(b)
    return 2*abs(a-b) / float(a+b)

def gradDescent(X, Y, W0, b0, momentum=False, out=True):
    i = 0
    if momentum:
        Z = W0.copy()
        v = b0.copy()
    W = W0.copy()
    b = b0.copy()
    WPrev = W0 - 10*EPS
    bPrev = b0 - 10*EPS
    while (norm(W-WPrev)>EPS or norm(b-bPrev)>EPS) and i<MAX_ITER:
        WPrev = W.copy()
        bPrev = b.copy()
        P = forward(X, W, b)
        if momentum:
            Z = GAMMA * Z + ALPHA * dC_weight(X, Y, P)
            v = GAMMA * v + ALPHA * dC_bias(X, Y, P)
            W -= Z
            b -= v
        else:
            W -= ALPHA * dC_weight(X, Y, P)
            b -= ALPHA * dC_bias(X, Y, P)
        if i % 500 == 0 and out:
            print "Iter", i
            print "C(Y, P) =", C(Y, P), "\n"
        i += 1
    return W, b

def genPath(X, Y, W0, b0, i, j, p, q, momentum=False):
    MR = 0.5
    LR = 1e-4 if momentum else 0.1
    W = W0.copy()
    b = b0.copy()
    path = [(1.4, 1.4)]
    W[i][j], W[p][q] = path[0]
    if momentum:
        Z = W.copy()
        v = b.copy()
    for k in range(20):
        P = forward(X, W, b)
        if momentum:
            Z[i][j] = MR*Z[i][j] + LR*dC_weight(X, Y, P)[i][j]
            Z[p][q] = MR*Z[p][q] + LR*dC_weight(X, Y, P)[p][q]
            W[i][j] -= Z[i][j]
            W[p][q] -= Z[p][q]
            v = MR*v + LR*dC_bias(X, Y, P)
            b -= v
        else:
            W[i][j] -= LR*dC_weight(X, Y, P)[i][j]
            W[p][q] -= LR*dC_weight(X, Y, P)[p][q]
            b -= LR*dC_bias(X, Y, P)
        path.append([W[i][j], W[p][q]])
    return path
