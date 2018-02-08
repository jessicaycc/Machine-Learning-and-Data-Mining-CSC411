from const import *
from getdata import *
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt


#______________________________________________ INITIALIZE _________________________________________________#
act = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon", "Alec Baldwin", "Bill Hader", "Steve Carell", "Kristin Chenoweth", "Fran Drescher", "America Ferrera", "Daniel Radcliffe", "Gerard Butler", "Michael Vartan"]

#getData(act)

trainSet, validSet, testSet = getSets(act, setRatio=(150, 0, 0))


#___________________________________________ DISTANCE FUNCTIONS ____________________________________________#
def euclidean(a, b):
    return norm(a-b)

def linfinity(a, b):
    return max(abs(a-b))

def lzero(a, b):
    return 0

def negCosine(a, b):
    return -dot(a, b) / (norm(a)*norm(b))


#_____________________________________________ GET NEIGHBOURS ______________________________________________#
def genX(imgSet):
    A = np.empty((0, VEC_SIZE), float)
    fileList = np.asarray(imgSet).flatten()
    for filename in fileList:
        img = Image.open("processed/" + filename)
        a = np.array(img).flatten() / 255.
        A = np.vstack((A, a))
    return A

def genY(M):
    alec = np.tile([1, 0], (M, 1))
    male = np.tile([0, 1], (5*M, 1))
    return np.vstack(( alec, male ))

def getNeighbours(X, Y, z, k):
    dist = np.array([ euclidean(z, x) for x in X ])
    kSmallest = np.argpartition(dist, k)[:k].tolist()
    return Y[kSmallest].tolist()


#____________________________________________ REPORT ACCURACY ______________________________________________#
def classify(Y):
    votes = {}
    for y in Y:
        vote = tuple(y)
        if vote in votes:
            votes[vote] += 1
        else:
            votes[vote] = 1
	sortVotes = sorted(votes, key=votes.__getitem__, reverse=True)
    return list(sortVotes[0])

def accuracy(X, Y, Z, W, k):
    correct = 0
    N = len(Z)
    for i in xrange(N):
        neighbours = getNeighbours(X, Y, Z[i], k)
        if not norm(classify(neighbours) - W[i]):
            correct += 1
    acc = correct / float(N)
    print acc
    return acc


#_________________________________________________ PLOT ____________________________________________________#
def plot(f, x, filename):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    y = np.vectorize(f, otypes=[float])(x)
    plt.plot(x, y)
    plt.savefig("plots/" + filename + ".png", bbox_inches="tight")
    plt.show()
    return


#__________________________________________________ MAIN ___________________________________________________#
X = genX(trainSet)
Y = genY(len(trainSet[0]))
k = np.arange(1, 21)*25



def f(x):
    return accuracy(X, Y, X, Y, x)

plot(f, k, "test")

quit()
