from const import *

def softmax(O):
    return exp(O)/tile(sum(exp(O),0), (len(O),1))

def forward(X, W, b):
    i = np.ones(len(X))
    return softmax(dot(X, W.T) + dot(i, b.T))

def genX():
    return 0

def genY():
    return 0