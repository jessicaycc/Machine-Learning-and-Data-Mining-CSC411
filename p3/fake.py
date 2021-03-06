import graphviz
import numpy as np
from const import *
from bayes import *
from getdata import *
from logistic import *
from sklearn import tree
from sklearn.metrics import accuracy_score

#______________________________ PART 1 ______________________________#
def part1():
    train, valid, test = (a+b for a,b in zip(genSets('clean_real.txt'), genSets('clean_fake.txt')))

    vocab   = genVocab(train)
    train_x = genX(train, vocab)
    valid_x = genX(valid, vocab)
    test_x  = genX(test,  vocab)
    train_y = genY('train')
    valid_y = genY('valid')
    test_y  = genY('test' )

    saveObj(vocab, 'vocab')
    saveObj(train_x, 'train_x')
    saveObj(valid_x, 'valid_x')
    saveObj(test_x,  'test_x' )
    saveObj(train_y, 'train_y')
    saveObj(valid_y, 'valid_y')
    saveObj(test_y,  'test_y' )

    train_x = loadObj('train_x')
    vocab = loadObj('vocab')
    vocab = list(vocab.keys())

    realSize = int(NUM_REAL*SET_RATIO[0])
    fakeSize = int(NUM_FAKE*SET_RATIO[0])

    real_x = np.sum(train_x[:realSize], axis=0) / realSize
    fake_x = np.sum(train_x[realSize:], axis=0) / fakeSize
    
    diff = abs(real_x-fake_x)
    index = np.argsort(diff)

    words = [vocab[i] for i in index]
    real = [real_x[i] for i in index]
    fake = [fake_x[i] for i in index]

    w = words[::-1][:3]
    r = real[::-1][:3]
    f = fake[::-1][:3]

    print("Three Keywords:")
    print(w)
    print(r)
    print(f)
    return

#______________________________ PART 2 ______________________________#
def part2():
    train_x = loadObj('train_x')
    valid_x = loadObj('valid_x')
    test_x  = loadObj('test_x')

    #naiveBayesGridSearch(train_x, valid_x, trainSet=False)

    print('Accuracy on train set: %.2f%%' % naiveBayes(train_x, train_x, trainSet=True))
    print('Accuracy on valid set: %.2f%%' % naiveBayes(train_x, valid_x, trainSet=False))
    print('Accuracy on test set: %.2f%%'  % naiveBayes(train_x, test_x,  trainSet=False))
    return

#______________________________ PART 3 ______________________________#
def part3():
    train_x = loadObj('train_x')
    
    m = 7
    p = 0.03

    realSize = int(NUM_REAL*SET_RATIO[0])
    fakeSize = int(NUM_FAKE*SET_RATIO[0])
    pReal = realSize / (realSize+fakeSize)
    pFake = fakeSize / (realSize+fakeSize)
    invertTrain_x = -1*(train_x - 1)

    real_x1 = ((np.sum(train_x[:realSize], axis=0)) + m*p) / (realSize+m)
    fake_x1 = ((np.sum(train_x[realSize:], axis=0)) + m*p) / (fakeSize+m)
    real_x0 = ((np.sum(invertTrain_x[:realSize], axis=0)) + m*p) / (realSize+m)
    fake_x0 = ((np.sum(invertTrain_x[realSize:], axis=0)) + m*p) / (fakeSize+m)

    real_presence = np.divide(list(map(lambda x: x * pReal, real_x1)), (real_x1*pReal + fake_x1*pFake))
    real_absence  = np.divide(list(map(lambda x: x * pReal, real_x0)), (real_x0*pReal + fake_x0*pFake))
    fake_presence = np.divide(list(map(lambda x: x * pFake, fake_x1)), (real_x1*pReal + fake_x1*pFake))
    fake_absence  = np.divide(list(map(lambda x: x * pFake, fake_x0)), (real_x0*pReal + fake_x0*pFake))

    print('TOP 10 WORDS:')
    getTop10(real_presence, 10)
    getTop10(real_absence,  10) 
    getTop10(fake_presence, 10) 
    getTop10(fake_absence,  10)

    print('\nTOP 10 NON-STOPWORDS:')
    getTop10_noStop(real_presence, 10)
    getTop10_noStop(real_absence,  10) 
    getTop10_noStop(fake_presence, 10) 
    getTop10_noStop(fake_absence,  10)
    return

#______________________________ PART 4 ______________________________#
def part4():
    vocab = loadObj('vocab')

    model = train(
        model=LogisticRegression(len(vocab)),
        loss_fn=nn.BCELoss(size_average=True),
        num_epochs=80,
        batch_size=24,
        learn_rate=1e-3,
        reg_rate=1e-4)

    saveObj(model, 'model')

    print('Accuracy on train set: %.2f%%' % test(model,'train'))
    print('Accuracy on valid set: %.2f%%' % test(model,'valid'))
    print('Accuracy on test set: %.2f%%' % test(model,'test'))
    return

#______________________________ PART 6 ______________________________#
def part6():
    vocab = loadObj('vocab')
    model = loadObj('model')

    vocab = list(vocab.keys())
    W = model.features[1].weight.data.numpy()[0]
    W_index_sorted = W.argsort()

    W_pos = W_index_sorted[-10:][::-1]
    W_neg = W_index_sorted[:10]

    top10_pos = [(W[i], vocab[i]) for i in W_pos]
    top10_neg = [(W[i], vocab[i]) for i in W_neg]

    print('Top 10 positive weights:', top10_pos)
    print('\nTop 10 negative weights:', top10_neg)
    
    W_pos = W_index_sorted[::-1]
    W_neg = W_index_sorted[:]

    top10_pos = [(W[i], vocab[i]) for i in W_pos if vocab[i] not in ENGLISH_STOP_WORDS][:10]
    top10_neg = [(W[i], vocab[i]) for i in W_neg if vocab[i] not in ENGLISH_STOP_WORDS][:10]

    print('\nTop 10 positive weights (no stop words):', top10_pos)
    print('\nTop 10 negative weights (no stop words):', top10_neg)
    return

#______________________________ PART 7 ______________________________#
def part7():
    vocab = loadObj('vocab')
    train_x = loadObj('train_x')
    train_y = loadObj('train_y')
    valid_x = loadObj('valid_x')
    valid_y = loadObj('valid_y')
    test_x  = loadObj('test_x')
    test_y  = loadObj('test_y')

    train_acc = list()
    valid_acc = list()
    depths = np.arange(1, 152, 10)

    dtc = tree.DecisionTreeClassifier(
        criterion='entropy',
        max_features=310,
        max_depth=151)

    dtc.fit(train_x, train_y)

    pred = dtc.predict(train_x)
    acc = accuracy_score(train_y, pred)*100
    print('Accuracy on train set: %.2f%%' % acc)

    pred = dtc.predict(valid_x)
    acc = accuracy_score(valid_y, pred)*100
    print('Accuracy on valid set: %.2f%%' % acc)

    pred = dtc.predict(test_x)
    acc = accuracy_score(test_y, pred)*100
    print('Accuracy on test set: %.2f%%' % acc)

    dot_data = tree.export_graphviz(
        dtc,
        out_file=None,
        feature_names=list(vocab.keys()),
        class_names=('real','fake'),
        max_depth=2,
        filled=True,
        rounded=True,
        special_characters=True)

    graph = graphviz.Source(dot_data) 
    graph.format = 'png'
    graph.render('plots/graph', view=True)

    for depth in depths:
        dtc = tree.DecisionTreeClassifier(
            criterion='entropy',
            max_depth=depth)

        dtc.fit(train_x, train_y)

        pred = dtc.predict(train_x)
        train_acc.append(accuracy_score(train_y, pred)*100)

        pred = dtc.predict(valid_x)
        valid_acc.append(accuracy_score(valid_y, pred)*100)

        print("Depth [{}/{}]: done".format(depth, depths[-1]))
    
    linegraph(train_acc, valid_acc, depths, 'curve_tree', 'Depth')
    return

#______________________________ PART 8 ______________________________#
def part8():
    def I(vocab, word, x):
        real_x1 = 0
        real_x0 = 0
        fake_x1 = 0
        fake_x0 = 0

        total = float(len(x))
        midpoint = int(NUM_REAL*SET_RATIO[0])
        vocab = list(vocab.keys())
        index = vocab.index(word)

        for i, n in enumerate(x):
            if i < midpoint: 
                if n[index] == 1:
                    real_x1 += 1
                else:
                    real_x0 += 1
            else:
                if n[index] == 1:
                    fake_x1 += 1
                else: 
                    fake_x0 += 1

        h_Y = H(real_x1 + real_x0, fake_x1 + fake_x0)
        h_YXi = (real_x1 + fake_x1)/total * H(real_x1, fake_x1) + (real_x0+fake_x0)/total * H(real_x0, fake_x0)
        mutualInfo = h_Y - h_YXi

        print(mutualInfo)
        return (mutualInfo)

    def H(x1, x2):
        total = float(x1 + x2)
        return -x1/total * np.log2(x1/total)-x2/total * np.log2(x2/total)

    vocab = loadObj('vocab')
    train_x = loadObj('train_x')    
    I(vocab, "donald", train_x)
    I(vocab, "star", train_x)
    return

#_______________________________ MAIN _______________________________#
if __name__ == '__main__':
    start = time.time()

    part1()
    part2()
    part3()
    part4()
    part6()
    part7()
    part8()

    end = time.time()
    print('Time elapsed: %.2fs' % (end-start))
