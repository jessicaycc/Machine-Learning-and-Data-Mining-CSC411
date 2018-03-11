import matplotlib.pyplot as plt
from const import *
from matplotlib import cm
from matplotlib.pyplot import clabel

if not os.path.exists('plots'):
    os.makedirs('plots')

def heatmap(x, shape, filename):
    img = np.reshape(x, shape)
    plt.imshow(img, cmap=cm.coolwarm)
    plt.savefig('plots/'+filename+'.png', bbox_inches='tight')
    plt.show()
    return

def linegraph(y1, y2, x, filename):
    plt.plot(x, y1, label="validation")
    plt.plot(x, y2, label="training")
    plt.legend(loc="lower left")
    plt.savefig('plots/'+filename+'.png', bbox_inches='tight')
    plt.show()
    return

def contour(y, x, M, filename):
    cs = plt.contour(x, y, M)
    clabel(cs, inline=1, fontsize=10)
    plt.xlabel("w1")
    plt.ylabel("w2")
    plt.legend(loc='lower left')
    plt.savefig('plots/'+filename+'.png', bbox_inches='tight')
    plt.show()
    return
