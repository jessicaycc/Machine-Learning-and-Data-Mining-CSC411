from const import *

def genSets(input, ratio=DATA_SET_RATIO):
    shuffled = list()
    with open(input) as f:
        for line in f:
            shuffled.append(line.split())
    np.random.shuffle(shuffled)
    size = len(shuffled)
    set1 = shuffled[:int(size*ratio[0])]
    set2 = shuffled[int(size*ratio[0]):int(size*(ratio[0]+ratio[1]))]
    set3 = shuffled[int(size*(ratio[0]+ratio[1])):]
    return set1, set2, set3

def genVocab(input):
    vocab = list()
    for line in input:
        for word in line:
            if word not in vocab:
                vocab.append(word)
    vocab = sorted(vocab)
    return {k: v for v,k in enumerate(vocab)}

def genX(input, vocab):
    X = np.zeros((len(input), len(vocab)))
    for i, line in enumerate(input):
        for word in line:
            if word in vocab:
                X[i][vocab[word]] = 1.
    return X
