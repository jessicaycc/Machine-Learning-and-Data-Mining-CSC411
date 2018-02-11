from const import *

def softmax(O):
    return exp(O)/np.tile(sum(exp(O),0), (len(O),1))

def forward(X, W, b):
    '''
    i = np.ones((len(X), 1))
    print "forward", softmax(dot(X, W.T) + dot(i, b.T))
    return softmax(dot(X, W.T) + dot(i, b.T))
    '''
    L = np.matmul(W, X.T) + b
    p = softmax(L) #10x1000
    return p
def genX(M):
    X = np.empty((0, NUM_FEAT), float)
    for i in range(NUM_LABEL):
        filename = 'train' + str(i)
        for j in range(SET_RATIO[0]):
            x = M[filename][j].reshape(IMG_SHAPE).flatten() / 255.
            X = np.vstack((X, x))
    return X

def genY(n):
    Y = np.empty((0, NUM_LABEL), float)
    for i in range(NUM_LABEL):
        Y = np.vstack(( Y, np.tile(LABEL[i], (n, 1)) ))
    return Y
