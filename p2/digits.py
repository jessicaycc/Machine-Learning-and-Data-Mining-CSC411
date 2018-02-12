from calc import *
from plot import *
from scipy.io import loadmat

np.random.seed(0)
M = loadmat("mnist_all.mat")

#______________________________________________ PART 3 ______________________________________________#
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

#______________________________________________ PART 4 ______________________________________________#
def part4():
    def f(n):
        X = genX(M, TRAIN, n)
        Y = genY(n)
        W = np.zeros((NUM_LABEL, NUM_FEAT))
        b = np.zeros((NUM_LABEL, 1))

        W, b = gradDescent(X, Y, W, b, out=False)

        X = genX(M, TEST, 100)
        Y = genY(100)
        P = classify(X, Y, W, b)
        res = accuracy(P, Y)

        print "({}, {}) - point generated".format(n, res)
        return res
    
    X = genX(M, TRAIN, 100)
    Y = genY(100)
    W = np.zeros((NUM_LABEL, NUM_FEAT))
    b = np.zeros((NUM_LABEL, 1))

    W, b = loadObj("weights"), loadObj("bias")
    #W, b = gradDescent(X, Y, W, b)
    #saveObj(W, "weights")
    #saveObj(b, "bias")

    for i in range(len(W)):
        heatmap(W[i], "pt4_weight_" + str(i))

    linegraph(f, np.arange(1, 11)*10, "pt4_learning_curve")
    return

#______________________________________________ PART 5 ______________________________________________#
def part5():
    X = genX(M, TRAIN, 100)
    Y = genY(100)

    W = np.zeros((NUM_LABEL, NUM_FEAT))
    b = np.zeros((NUM_LABEL, 1))
    W, b = gradDescent(X, Y, W, b)
    
    W = np.zeros((NUM_LABEL, NUM_FEAT))
    b = np.zeros((NUM_LABEL, 1))
    W, b = gradDescent(X, Y, W, b, momentum=True)

    return

#_______________________________________________ MAIN _______________________________________________#

#part3()
#part4()
#part5()
