from const import *

def softmax(O):
    return exp(O)/np.tile(sum(exp(O),0), (len(O),1))

def forward(X, W, b):
    i = np.ones(len(X))
    return softmax(dot(X, W.T) + dot(i, b.T))

def genX():
    return 0

def genY(n):
    Y = np.empty((0, 10), float)
    for i in range(10):
        Y = np.vstack(( Y, np.tile(LABEL[i], (n, 1)) ))
    return Y