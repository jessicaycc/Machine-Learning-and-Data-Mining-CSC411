import time
from calc import *
from plot import *
from scipy.io import loadmat

np.random.seed(0)
M = loadmat("mnist_all.mat")

#______________________________ PART 3 ______________________________#
def part3():
    X = genX(M, TRAIN, 100)
    Y = genY(100)
    W = np.random.rand(NUM_LABEL, NUM_FEAT)
    b = np.random.rand(NUM_LABEL, 1)
    P = forward(X, W, b)

    weightErr = 0.
    biasErr = 0.

    k = 0.
    grad = dC_weight(X, Y, P)
    while k < 10.:
        i = np.random.randint(0, NUM_LABEL)
        j = np.random.randint(0, NUM_FEAT)
        if grad[i][j]:
            diff = finiteDiff_weight(X, Y, W, b, i, j)
            weightErr += relativeError(grad[i][j], diff)
            k += 1.
        else:
            continue

    print "Weight gradient relative error:", weightErr/k
    
    k = 0.
    grad = dC_bias(X, Y, P)
    for i in range(NUM_LABEL):
        if grad[i]:
            diff = finiteDiff_bias(X, Y, W, b, i)
            biasErr += relativeError(grad[i][0], diff)
            k += 1.
    
    print "Bias gradient relative error:", biasErr/k
    return

#______________________________ PART 4 ______________________________#
def part4():
    def f(n):
        X = genX(M, TRAIN, n)
        Y = genY(n)
        W = np.zeros((NUM_LABEL, NUM_FEAT))
        b = np.zeros((NUM_LABEL, 1))

        W, b = gradDescent(X, Y, W, b, out=False)

        X = genX(M, TEST, MAX_TEST_SIZE)
        Y = genY(MAX_TEST_SIZE)
        P = classify(X, Y, W, b)
        res = accuracy(P, Y)

        print "({}, {}) - point generated".format(n, res)
        return res
    
    X = genX(M, TRAIN, MAX_TRAIN_SIZE)
    Y = genY(MAX_TRAIN_SIZE)
    W = np.zeros((NUM_LABEL, NUM_FEAT))
    b = np.zeros((NUM_LABEL, 1))

    W, b = gradDescent(X, Y, W, b)
    #W, b = loadObj("weights"), loadObj("bias")
    #saveObj(W, "weights")
    #saveObj(b, "bias")

    for i in range(len(W)):
        heatmap(W[i], "pt4_weight_" + str(i))

    x = np.arange(0, MAX_TRAIN_SIZE, 100)
    linegraph(f, x, "pt4_learning_curve")
    return

#______________________________ PART 5 ______________________________#
def part5():
    def f(n):
        X = genX(M, TRAIN, n)
        Y = genY(n)
        W = np.zeros((NUM_LABEL, NUM_FEAT))
        b = np.zeros((NUM_LABEL, 1))

        W, b = gradDescent(X, Y, W, b, momentum=True, out=False)

        X = genX(M, TEST, MAX_TEST_SIZE)
        Y = genY(MAX_TEST_SIZE)
        P = classify(X, Y, W, b)
        res = accuracy(P, Y)

        print "({}, {}) - point generated".format(n, res)
        return res

    x = np.arange(0, MAX_TRAIN_SIZE, 100)
    linegraph(f, x, "pt5_learning_curve")
    return

#______________________________ PART 6 ______________________________#
def part6a():
    X = genX(M, TRAIN, 500)
    Y = genY(500)
    W = loadObj("weights")
    b = loadObj("bias")

    w1s = np.arange(-0.1, 0.1, 0.01)
    w2s = np.arange(-0.1, 0.1, 0.01)
    w1z, w2z = np.meshgrid(w1s, w2s)

    cost = np.zeros((w1s.size, w2s.size))
    for i, w1 in enumerate(w1s):
        for j, w2 in enumerate(w2s):
            W[5][150] = w1
            W[6][150] = w2
            P = forward(X, W, b)
            cost[i][j] = C(Y, P)
            
    contour(w1z, w2z, cost, "pt6_contour")
    return

def part6b():
    return

def part6c():
    return

#______________________________ PART 7 ______________________________#
def part7():
    return


#_______________________________ MAIN _______________________________#

if __name__ == "__main__":
    start = time.time()

    #part3()
    #part4()
    #part5()
    #part6a()
    #part6b()
    #part6c()
    #part7()

    end = time.time()
    print "Time elapsed:", end-start
