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
            Z = GAMMA * Z + ALPHA_M * dC_weight(X, Y, P) 
            v = GAMMA * v + ALPHA_M * dC_bias(X, Y, P)
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



# Extra code
#def randomize(A, B):
#    permutation = np.random.permutation(A.shape[0])
#    Arand = A[permutation]
#    Brand = B[permutation]
#    return Arand, Brand
#
#def gradDescent_MB(X, Y, W, b, batchSize=128):
#    Z = W.copy()
#    v = b.copy()
#    for iter in range(MAX_ITER):
#        Xrand, Yrand = randomize(X, Y)
#        for i in range(0, Xrand.shape[0], batchSize):
#            Xsub = Xrand[i:i + batchSize]
#            Ysub = Yrand[i:i + batchSize]
#            P = forward(Xsub, W, b)
#            W -= ALPHA * dC_weight(Xsub, Ysub, P) 
#            b -= ALPHA * dC_bias(Xsub, Ysub, P)
#        if iter % 500 == 0:
#            print "Iter", iter
#            print "C(Y, P) =", C(Ysub, P), "\n"
#    return W, b
