from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt

act = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon", "Alec Baldwin", "Bill Hader", "Steve Carell"]
trainSet, validSet, testSet= getSets(act)
def getTrian():
    trainSet_X = trainSet
    trainSet_Y = np.empty((k, 0), float)
    for i in range(0, k):
        trainSet_Y = np.hstack( (trainSet_Y, np.tile(I[:,[i]], (1, N))) )
    return trainSet_X, trainSet_Y

def getTest():
    testSet_X = testSet 
    testSet_Y = np.empty((k, 0), float)
    for i in range(0, k):
        testSet_Y = np.hstack( (testSet_Y, np.tile(I[:,[i]], (1, N))) )
