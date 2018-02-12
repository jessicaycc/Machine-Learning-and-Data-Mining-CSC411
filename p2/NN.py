import os
import cPickle
from const import *

def save(obj, filename):
    if not os.path.exists('objects'):
        os.makedirs('objects')
    cPickle.dump(obj, open('objects/'+filename+'.pk', 'wb'))
    return

def load(filename):
    return cPickle.load(open('objects/'+filename+'.pk', 'rb'))

def softmax(O):
    return exp(O)/exp(O).sum(axis=1, keepdims=True)

def forward(X, W, b):
    i = np.ones((len(X), 1))                 
    return softmax(dot(X, W.T) + dot(i, b.T))    #dimension same as Y

def genX(M, set, size):
    X = np.empty((0, NUM_FEAT), float)
    for i in range(NUM_LABEL):
        filename = set + str(i)
        for j in range(size):
            x = M[filename][j].reshape(IMG_SHAPE).flatten() / 255.
            X = np.vstack((X, x))
    return X

def genY(size):
    Y = np.empty((0, NUM_LABEL), float)
    for i in range(NUM_LABEL):
        Y = np.vstack(( Y, np.tile(LABEL[i], (size, 1)) ))
    return Y

def classify():
    return

def accuracy():
    return
