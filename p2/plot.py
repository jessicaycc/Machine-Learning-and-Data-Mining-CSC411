import matplotlib.pyplot as plt

def linegraph(f, x, filename):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    y = np.vectorize(f, otypes=[float])(x)
    plt.plot(x, y)
    plt.savefig('plots/'+filename+'.png', bbox_inches='tight')
    plt.show()
    return