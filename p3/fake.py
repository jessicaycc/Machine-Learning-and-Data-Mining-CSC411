import math
import operator
import numpy as np
from functools import reduce

from getdata import *
#______________________________ PART 1 ______________________________#
def part1():
    train, valid, test = (a+b for a,b in zip(genSets('clean_real.txt'), genSets('clean_fake.txt')))

    vocab = genVocab(train)
    train_x = genX(train, vocab)
    valid_x = genX(valid, vocab)
    test_x = genX(test, vocab)

    saveObj(vocab, "vocab")
    saveObj(train_x, "train_x")
    saveObj(valid_x, "valid_x")
    saveObj(test_x, "test_x")
    return

#______________________________ PART 2 ______________________________#
def part2():
    train_x = loadObj("train_x")
    valid_x = loadObj("valid_x")
    naiveBayes(train_x, valid_x, trainSet = True)
    naiveBayes(train_x,valid_x, trainSet = False)
    return

#______________________________ PART 3 ______________________________#
def part3():
    train_x = loadObj("train_x")
    
    m = 7
    p = 0.03
    realSize = int(1968*DATA_SET_RATIO[0])
    fakeSize = int(1298*DATA_SET_RATIO[0])
    pReal = realSize/(realSize+fakeSize)
    pFake = fakeSize/(realSize+fakeSize)
    invertTrain_x = -1*(train_x - 1)
    real_x1 = ((np.sum(train_x[:realSize],axis=0))+m*p)/(realSize+m)
    fake_x1 = ((np.sum(train_x[realSize:],axis=0))+m*p)/(fakeSize+m)
    real_x0 = ((np.sum(invertTrain_x[:realSize],axis=0))+m*p)/(realSize+m)
    fake_x0 = ((np.sum(invertTrain_x[realSize:],axis=0))+m*p)/(fakeSize+m)
    real_presence = np.divide(list(map(lambda x: x * pReal, real_x1)), (real_x1*pReal+fake_x1*pFake))
    real_absence = np.divide(list(map(lambda x: x * pReal, real_x0)), (real_x0*pReal+fake_x0*pFake))
    fake_presence = np.divide(list(map(lambda x: x * pFake, fake_x1)), (real_x1*pReal+fake_x1*pFake))
    fake_absence = np.divide(list(map(lambda x: x * pFake, fake_x0)), (real_x0*pReal+fake_x0*pFake))
    print ("TOP 10 WORDS:")
    getTop10(real_presence,10)
    getTop10(real_absence,10) 
    getTop10(fake_presence,10) 
    getTop10(fake_absence,10)
    print ("\nTOP 10 NON-STOPWORDS:")
    getTop10_noStop(real_presence,10)
    getTop10_noStop(real_absence,10) 
    getTop10_noStop(fake_presence,10) 
    getTop10_noStop(fake_absence,10)

    return

#______________________________ PART 4 ______________________________#
def part4():
    return

#______________________________ PART 5 ______________________________#
def part5():
    return

#______________________________ PART 6 ______________________________#
def part6():
    return

#______________________________ PART 7 ______________________________#
def part7():
    return

#______________________________ PART 8 ______________________________#
def part8():
    return

#_______________________________ MAIN _______________________________#
if __name__ == '__main__':
    start = time.time()

    #part1()
    #part2()
    part3()
    #part4()
    #part5()
    #part6()
    #part7()
    #part8()
  
    end = time.time()
    print('Time elapsed:', end-start)
