from calc import *
from plot import *
from scipy.io import loadmat

np.random.seed(0)
M = loadmat('mnist_all.mat')

#______________________________________________ PART 3 ______________________________________________#
def part3():
    X = genX(M, TRAIN, 100)
    Y = genY(100)
    W = np.random.rand(NUM_LABEL, NUM_FEAT)
    b = np.random.rand(NUM_LABEL, 1)
    P = forward(X, W, b)

    n = dC_weight(X, Y, P)
    m = finiteDiff_weight(X, Y, W, b, 5, 157)
    print relativeError(n[5][157], m)

    n = dC_bias(X, Y, P)
    m = finiteDiff_bias(X, Y, W, b, 1)
    print relativeError(n[1][0], m)
    return

#______________________________________________ PART 4 ______________________________________________#
def part4():
    def f(n):
        X = genX(M, TRAIN, n)
        Y = genY(n)
        W = np.zeros((NUM_LABEL, NUM_FEAT))
        b = np.zeros((NUM_LABEL, 1))

        W, b = gradDescent(X, Y, W, b)

        X = genX(M, TEST, 100)
        res = accuracy(X, W, b)

        print "({}, {}) - point generated".format(n, res)
        return res
    
    linegraph(f, np.arange(1, 11)*10, 'pt4')
    return

#______________________________________________ PART 5 ______________________________________________#
def part5():
    return

#_______________________________________________ MAIN _______________________________________________#

#part3()
part4()
#part5()
