from const import *
from calc import *
from NN import *
from scipy.io import loadmat

np.random.seed(0)
M = loadmat("mnist_all.mat")
print np.shape(M)

def genX():
    Matrix = np.empty((0,784), float)
    for i in range (0,10):
        for j in range (SET_RATIO[0]):
            filename = "train" + str(i)
            x =np.array([ M[filename][j].reshape(IMG_SHAPE).flatten() / 255. ])
            Matrix = np.vstack((Matrix, x))
    return Matrix
X = genX()
#Y = genY(M)
X = genX(M)
Y = genY(SET_RATIO[0])

#x = np.array([ M["train0"][150].reshape(IMG_SHAPE).flatten() / 255. ]).T
#y = np.array([ [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] ]).T
