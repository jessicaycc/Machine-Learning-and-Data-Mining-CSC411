import os
import matplotlib.pyplot as plt
from const import *
from matplotlib import cm
from matplotlib.pyplot import clabel

if not os.path.exists("plots"):
    os.makedirs("plots")

def heatmap(x, filename):
    img = np.reshape(x, IMG_SHAPE)
    plt.imshow(img, cmap=cm.coolwarm)
    plt.savefig("plots/"+filename+".png", bbox_inches="tight")
    plt.show()
    return

def linegraph(f, x, filename):
    y = np.vectorize(f, otypes=[float])(x)
    plt.plot(x, y)
    plt.savefig("plots/"+filename+".png", bbox_inches="tight")
    plt.show()
    return

def contour(x, y, M, filename):
    cs = plt.contour(x, y, M)
    clabel(cs, inline=1, fontsize=10)
    plt.savefig("plots/"+filename+".png", bbox_inches="tight")
    plt.show()
    return
