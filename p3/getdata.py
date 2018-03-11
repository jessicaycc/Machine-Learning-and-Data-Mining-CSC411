import math
import operator
import numpy as np
from functools import reduce
from const import *

def genSets(input, ratio=SET_RATIO):
    shuffled = list()
    with open(input) as f:
        for line in f:
            shuffled.append(line.split())
    np.random.shuffle(shuffled)
    size = [int(x*len(shuffled)) for x in ratio]
    set1 = shuffled[:size[0]]
    set2 = shuffled[size[0]:size[0]+size[1]]
    set3 = shuffled[size[0]+size[1]:sum(size)]
    return set1, set2, set3

def genVocab(set):
    vocab = list()
    for line in set:
        for word in line:
            if word not in vocab:
                vocab.append(word)
    vocab = sorted(vocab)
    return {k: v for v,k in enumerate(vocab)}

def genX(set, vocab):
    X = np.zeros((len(set), len(vocab)))
    for i, line in enumerate(set):
        for word in line:
            if word in vocab:
                X[i][vocab[word]] = 1.
    return X

def naiveBayes(train_x, valid_x, trainSet = True):
    count = 0
    m = 7
    p = 0.03
    realSize = int(1968*SET_RATIO[0])
    fakeSize = int(1298*SET_RATIO[0])
    pReal = realSize/(realSize+fakeSize)
    pFake = fakeSize/(realSize+fakeSize)
    real = ((np.sum(train_x[:realSize],axis=0))+m*p)/(realSize+m)
    fake = ((np.sum(train_x[realSize:],axis=0))+m*p)/(fakeSize+m)
    if trainSet:
        midpoint = int(1968*SET_RATIO[0])
        set = train_x
        total = 2285.0
    else:
        midpoint = int(1968*SET_RATIO[1])
        set = valid_x
        total = 489.0
    for j, line in enumerate(set):
        realLst = [real[i] for i, n in enumerate(line) if n == 1]
        realLst = list(map(lambda x: math.log10(x),realLst))
        realLst = exp(reduce(operator.add,realLst))
        fakeLst = [fake[i] for i, n in enumerate(line) if n == 1]
        fakeLst = list(map(lambda x: math.log10(x),fakeLst))
        fakeLst = exp(reduce(operator.add,fakeLst))
        predReal = realLst*pReal
        predFake = fakeLst*pFake
        pred = 1 if predFake > predReal else 0
        if j < midpoint:
            if pred ==0:
                count+=1
        else:
            if pred ==1:
                count+=1
    accuracy = (count/total)*100
    print ("Accuracy:", accuracy, "%")
    return accuracy

def getTop10(array, top):
    vocab = loadObj("vocab")
    vocab = list(vocab.keys())
    index = np.argpartition(array, -top)[-top:]
    index = index[np.argsort(array[index])]
    top10 = [vocab[i] for i in index] 
    print(top10)
    return top10

def getTop10_noStop(array, top):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    vocab = loadObj("vocab")
    vocab = list(vocab.keys())
    index = np.argsort(array)
    topWords = [vocab[i] for i in (index) if vocab[i] not in ENGLISH_STOP_WORDS]
    top10 = topWords[-top:]
    print (top10)
    return top10

def genY(set, ratio=SET_RATIO):
    size = {
        'train': ratio[0],
        'valid': ratio[1],
        'test' : ratio[2]
        }[set]
    real = np.zeros(int(size*NUM_REAL))
    fake = np.ones(int(size*NUM_FAKE))
    Y = np.concatenate((real, fake))
    return Y
