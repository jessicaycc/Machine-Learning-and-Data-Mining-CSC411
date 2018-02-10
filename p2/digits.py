from numpy import dot
from numpy import log

def softmax(o):
    return exp(o) / tile(sum(exp(o), 0), (len(o), 1))

def forward(x, W, b)
    return softmax(dot(W.T, x) + b)

def C(y, p)
    return -sum(y * log(p))

def dC(y, p)
    return p - y