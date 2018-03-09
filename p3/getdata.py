import numpy as np
from const import *
import random
def getSets(textFile, setRatio=DATA_SET_RATIO):
    set1, set2, set3 = list(), list(), list()
    dataSize = 1298 if textFile == "clean_fake.txt" else 1968
    shuffled = open(textFile).readlines()
    random.shuffle(shuffled)
    set1.append(shuffled[:int(dataSize*setRatio[0])])
    set2.append(shuffled[int(dataSize*setRatio[0]):int(dataSize*(setRatio[0]+setRatio[1]))])
    set3.append(shuffled[int(dataSize*(setRatio[0]+setRatio[1])):])
    return set1, set2, set3

getSets("clean_real.txt")