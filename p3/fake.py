from getdata import *
from logistic import *

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
    input_size = len(vocab)
    num_classes = 2

    model = train(
        model = LogisticRegression(input_size, num_classes),
        loss_fn = nn.CrossEntropyLoss(),
        num_epochs = 100,
        batch_size = 10,
        learn_rate = 1e-3)

    print('Accuracy on train set: %.2f%%' % test(model,'train'))
    print('Accuracy on test set: %.2f%%' % test(model,'test'))
    return

#______________________________ PART 5 ______________________________#
def part5():
    return

#______________________________ PART 6 ______________________________#
def part6():
    return

#______________________________ PART 7 ______________________________#
def part7():
    return

#______________________________ PART 8 ______________________________#
def part8():
    return

#_______________________________ MAIN _______________________________#
if __name__ == '__main__':
    start = time.time()

    #part1()
    #part2()
    #part3()
    part4()
    #part5()
    #part6()
    #part7()
    #part8()

    end = time.time()
    print('Time elapsed: %.4fs' % (end-start))
