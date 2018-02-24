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

def linegraphFunc(f, x, filename):
    y = np.vectorize(f, otypes=[float])(x)
    plt.plot(x, y)
    plt.savefig('plots/'+filename+'.png', bbox_inches='tight')
    plt.show()
    return

def linegraphVec(y1, y2, x, filename):
    plt.plot(x, y1, label="validation")
    plt.plot(x, y2, label = "training")
    plt.legend(loc="lower left")
    plt.savefig('plots/'+filename+'.png', bbox_inches='tight')
    plt.show()
    return

def contour(x, y, M, p1, p2, filename):
    cs = plt.contour(x, y, M)
    clabel(cs, inline=1, fontsize=10)
    plt.plot([a for a,b in p1], [b for a,b in p1], 'yo-', label='No Momentum')
    plt.plot([a for a,b in p2], [b for a,b in p2], 'ro-', label='Momentum')
    plt.xlabel("W1")
    plt.ylabel("W2")
    plt.legend(loc='lower left')
    plt.savefig('plots/'+filename+'.png', bbox_inches='tight')
    plt.show()
    return
