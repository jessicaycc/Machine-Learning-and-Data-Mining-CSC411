import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from const import *
from pylab import *

def heatmap(x, filename):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    M = np.reshape(x[:VEC_SIZE], DATA_SIZE)
    plt.imshow(M, cmap=cm.coolwarm)
    plt.savefig("plots/" + filename + ".png", bbox_inches="tight")
    plt.show()
    return

def linegraph(f, x, filename):
    if not os.path.exists("plots"):
        os.makedirs("plots")
    y = np.vectorize(f, otypes=[float])(x)
    plt.plot(x, y)
    plt.savefig("plots/" + filename + ".png", bbox_inches="tight")
    plt.show()
    return
