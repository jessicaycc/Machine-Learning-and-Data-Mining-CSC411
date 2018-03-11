from const import *
from getdata import *
from logistic import *
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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
    return

#______________________________ PART 2 ______________________________#
def part2():
    return

#______________________________ PART 3 ______________________________#
def part3():
    return

#______________________________ PART 4 ______________________________#
def part4():
    vocab = loadObj('vocab')

    model = train(
        model=LogisticRegression(len(vocab)),
        loss_fn=nn.BCELoss(size_average=True),
        num_epochs=50,
        batch_size=24,
        learn_rate=1e-3,
        reg_rate=1e-4)

    saveObj(model, 'model')

    print('Accuracy on train set: %.2f%%' % test(model,'train'))
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
    return

#_______________________________ MAIN _______________________________#
if __name__ == '__main__':
    start = time.time()

    #part1()
    #part2()
    #part3()
    #part4()
    part6()
    #part7()

    end = time.time()
    print('Time elapsed: %.2fs' % (end-start))
