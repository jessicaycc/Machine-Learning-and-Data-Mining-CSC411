import operator
import numpy as np
from const import *
from functools import reduce

def naiveBayesGridSearch(train_x, valid_x, trainSet=True):
    ref = 0.0
    refM = 0.0
    refP = 0.0

    for m in np.arange (1.0, 10.0, 1):
        for p in np.arange (0.01, 0.1, 0.01):
            count = 0

            realSize = int(NUM_REAL*SET_RATIO[0])
            fakeSize = int(NUM_FAKE*SET_RATIO[0])
            pReal = realSize / (realSize+fakeSize)
            pFake = fakeSize / (realSize+fakeSize)
            invertTrain_x = -1*(train_x - 1)

            real = ((np.sum(train_x[:realSize], axis=0)) + m*p) / (realSize+m)
            fake = ((np.sum(train_x[realSize:], axis=0)) + m*p) / (fakeSize+m)
            real_x0 = ((np.sum(invertTrain_x[:realSize], axis=0)) + m*p) / (realSize+m)
            fake_x0 = ((np.sum(invertTrain_x[realSize:], axis=0)) + m*p) / (fakeSize+m)

            if trainSet:
                midpoint = int(NUM_REAL*SET_RATIO[0])
                set = train_x
                total = 2285.0
            else:
                midpoint = int(NUM_REAL*SET_RATIO[1])
                set = valid_x
                total = 489.0

            for j, line in enumerate(set):
                realLst = []
                fakeLst = []

                for i, n in enumerate(line):
                    if n == 1:
                        realLst.append(real[i])
                        fakeLst.append(fake[i])
                    else:
                        realLst.append(real_x0[i])
                        fakeLst.append(fake_x0[i])

                realLst = list(map(lambda x: log(x), realLst))
                realLst = exp(reduce(operator.add, realLst))
                
                fakeLst = list(map(lambda x: log(x), fakeLst))
                fakeLst = exp(reduce(operator.add, fakeLst))

                predReal = realLst * pReal
                predFake = fakeLst * pFake
                pred = 1 if predFake > predReal else 0

                if j < midpoint:
                    if pred == 0:
                        count += 1
                else:
                    if pred == 1:
                        count += 1

            accuracy = 100 * count/total
            print (accuracy)
            print (m)
            print (p)

            if accuracy > ref:
                ref = accuracy
                refM = m
                refP = p
              
    print ("m", refM)
    print ("p", refP)
    return (ref)

def naiveBayes(train_x, valid_x, trainSet=True):
    m = 1.0
    p = 0.069
    count = 0

    realSize = int(NUM_REAL*SET_RATIO[0])
    fakeSize = int(NUM_FAKE*SET_RATIO[0])
    pReal = realSize / (realSize+fakeSize)
    pFake = fakeSize / (realSize+fakeSize)
    invertTrain_x = -1*(train_x - 1)

    real = ((np.sum(train_x[:realSize], axis=0)) + m*p) / (realSize+m)
    fake = ((np.sum(train_x[realSize:], axis=0)) + m*p) / (fakeSize+m)
    real_x0 = ((np.sum(invertTrain_x[:realSize], axis=0)) + m*p) / (realSize+m)
    fake_x0 = ((np.sum(invertTrain_x[realSize:], axis=0)) + m*p) / (fakeSize+m)

    if trainSet:
        midpoint = int(NUM_REAL*SET_RATIO[0])
        set = train_x
        total = 2285.0
    else:
        midpoint = int(NUM_REAL*SET_RATIO[1])
        set = valid_x
        total = 489.0

    for j, line in enumerate(set):
        realLst = []
        fakeLst = []

        for i, n in enumerate(line):
            if n == 1:
                realLst.append(real[i])
                fakeLst.append(fake[i])
            else:
                realLst.append(real_x0[i])
                fakeLst.append(fake_x0[i])

        realLst = list(map(lambda x: log(x), realLst))
        realLst = exp(reduce(operator.add, realLst))
        
        fakeLst = list(map(lambda x: log(x), fakeLst))
        fakeLst = exp(reduce(operator.add, fakeLst))

        predReal = realLst * pReal
        predFake = fakeLst * pFake
        pred = 1 if predFake > predReal else 0

        if j < midpoint:
            if pred == 0:
                count += 1
        else:
            if pred == 1:
                count += 1

    return (100 * count/total)
    
def getTop10(array, top):
    vocab = loadObj('vocab')
    vocab = list(vocab.keys())

    index = np.argsort(array)
    probs = [array[i] for i in index]
    probs = probs[::-1][:top]
    topWords = [vocab[i] for i in index]
    top10 = topWords[::-1][:top]

    print (probs)
    print(top10)
    return top10

def getTop10_noStop(array, top):
    vocab = loadObj('vocab')
    vocab = list(vocab.keys())

    index = np.argsort(array)
    topWords = [vocab[i] for i in index if vocab[i] not in ENGLISH_STOP_WORDS]
    top10 = topWords[::-1][:top]

    print(top10)
    return top10
