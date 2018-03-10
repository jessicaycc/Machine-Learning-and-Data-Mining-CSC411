from const import *

def genSets(input, ratio=SET_RATIO):
    shuffled = list()
    with open(input) as f:
        for line in f:
            shuffled.append(line.split())
    np.random.shuffle(shuffled)
    size = [int(x*len(shuffled)) for x in ratio]
    set1 = shuffled[:size[0]]
    set2 = shuffled[size[0]:size[0]+size[1]]
    set3 = shuffled[size[0]+size[1]:sum(size)]
    return set1, set2, set3

def genVocab(set):
    vocab = list()
    for line in set:
        for word in line:
            if word not in vocab:
                vocab.append(word)
    vocab = sorted(vocab)
    return {k: v for v,k in enumerate(vocab)}

def genX(set, vocab):
    X = np.zeros((len(set), len(vocab)))
    for i, line in enumerate(set):
        for word in line:
            if word in vocab:
                X[i][vocab[word]] = 1.
    return X

def genY(set, ratio=SET_RATIO):
    size = {
        'train': ratio[0],
        'valid': ratio[1],
        'test' : ratio[2]
        }[set]
    real = np.zeros(int(size*NUM_REAL))
    fake = np.ones(int(size*NUM_FAKE))
    Y = np.concatenate((real, fake))
    return Y
