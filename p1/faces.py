from plot import *
from const import *
from getdata import *
from regression import *

act = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon", "Alec Baldwin", "Bill Hader", "Steve Carell"]
act2 = ["Kristin Chenoweth", "Fran Drescher", "America Ferrera", "Daniel Radcliffe", "Gerard Butler", "Michael Vartan"]

#getData(act)
#getData(act2)

trainSet, validSet, testSet = getSets(act)
trainSet2, validSet2, testSet2 = getSets(act2)

N = len(trainSet[0])
M = len(validSet[0])

#______________________________________________ PART 3 ______________________________________________#
def part3():
    X = genMatrix(list(trainSet[ALEC]) + list(trainSet[STEV]))
    y = np.concatenate( (np.ones(N), np.zeros(N)) )
    w0 = np.zeros(VEC_SIZE + 1)
    w = gradDescent(Jbin, dJbin, X, w0, y)
    
    X = genMatrix(list(validSet[ALEC]) + list(validSet[STEV]))
    y = np.concatenate( (np.ones(M), np.zeros(M)) )

    print "Validation set cost =", Jbin(X, w, y)
    print "Training set accuracy =", accuracy(trainSet[ALEC::2], w, [1, 0])
    print "Validation set accuracy =", accuracy(validSet[ALEC::2], w, [1, 0])
    return

#______________________________________________ PART 4 ______________________________________________#
def part4():
    X = genMatrix(list(trainSet[ALEC]) + list(trainSet[STEV]))
    y = np.concatenate( (np.ones(N), np.zeros(N)) )
    w0 = np.zeros(VEC_SIZE + 1)
    w = gradDescent(Jbin, dJbin, X, w0, y)

    X_2 = genMatrix(list(trainSet[ALEC][:2]) + list(trainSet[STEV][:2]))
    y_2 = [1, 1, 0, 0]
    w0_2 = np.zeros(VEC_SIZE + 1)
    w_2 = gradDescent(Jbin, dJbin, X_2, w0_2, y_2)

    heatmap(w, "pt4_fullset")
    heatmap(w_2, "pt4_twoeach")

    #w0 = np.ones(VEC_SIZE + 1)
    #w0 = np.random.random(VEC_SIZE + 1)
    for i in range(1, 5):
        j = 10**i
        w = gradDescent(Jbin, dJbin, X, w0, y, maxIter=j, out=False)
        heatmap(w, "pt4_0_" + str(j))
        #heatmap(w, "pt4_1_" + str(j))
        #heatmap(w, "pt4_rand_" + str(j))
        print "Iter " + str(j) + ": heatmap generated"
    return

#______________________________________________ PART 5 ______________________________________________#
def part5():
    def f(n):
        trainSet, validSet, _ = getSets(act, False, setRatio=(n, 10, 0))
        #validSet, _, _ = getSets(act2, False, (150, 0, 0))
        X = genMatrix(np.asarray(trainSet).flatten())
        y = np.concatenate( (np.ones(3*n), np.zeros(3*n)) )

        w0 = np.zeros(VEC_SIZE + 1)
        w = gradDescent(Jbin, dJbin, X, w0, y, maxIter=1e6, out=False)
        y = [1, 1, 1, 0, 0, 0]
        res = accuracy(validSet, w, y)
        #res = accuracy(trainSet, w, y)
        print "({}, {}) - point generated".format(n, res)
        return res
    
    linegraph(f, np.arange(1, 15)*5, "pt5_inact")
    #linegraph(f, np.arange(1, 15)*5, "pt5_inact_train")
    #linegraph(f, np.arange(1, 12)*10, "pt5_notact")
    return

#______________________________________________ PART 6 ______________________________________________#
def part6():
    k = 3
    X = genMatrix(np.asarray(trainSet[::2]).flatten()).T
    I = np.identity(k)
    Y = np.empty((k, 0), float)
    for i in range(0, k):
        Y = np.hstack( (Y, np.tile(I[:,[i]], (1, N))) )
    W = np.random.random((VEC_SIZE + 1, k))

    grad = dJ(X, W, Y)
    res = 0
    for i in range(0, 5):
        p, q = np.random.randint(0, VEC_SIZE + 1), np.random.randint(0, k)
        approx = finiteDiff(J, X, W, Y, p, q)
        res += percentDiff(grad[p][q], approx)
    print "Percent difference =", res/5., "%"
    return

#______________________________________________ PART 7 ______________________________________________#
def part7():
    k = 6
    X = genMatrix(np.asarray(trainSet).flatten()).T
    I = np.identity(k)
    Y = np.empty((k, 0), float)
    for i in range(0, k):
        Y = np.hstack( (Y, np.tile(I[:,[i]], (1, N))) )
    w0 = np.zeros((VEC_SIZE + 1, k))
    W = gradDescent(J, dJ, X, w0, Y, 5e-6, 1e3, maxIter=10000)
    print "Training set accuracy =", accuracy(trainSet, W, I)
    print "Validation set accuracy =", accuracy(validSet, W, I)
    return W

#______________________________________________ PART 8 ______________________________________________#
def part8():
    W = part7()
    for i in xrange(len(W[0])):
        filename = act[i].replace(' ','_').lower()
        heatmap(W[:,i], "pt8_" + filename)
        print filename + ": heatmap generated"
    return

#_______________________________________________ MAIN _______________________________________________#
#Comment out code to test parts individually

#part3()
#part4()
#part5()
#part6()
#part7()
#part8()

quit()
