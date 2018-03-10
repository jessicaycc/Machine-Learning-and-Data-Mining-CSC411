from getdata import *

#______________________________ PART 1 ______________________________#
def part1():
    train, valid, test = (a+b for a,b in zip(genSets('clean_real.txt'), genSets('clean_fake.txt')))

    vocab = genVocab(train)
    train_x = genX(train, vocab)
    valid_x = genX(valid, vocab)
    test_x = genX(test, vocab)

    saveObj(vocab, "vocab")
    saveObj(train_x, "train_x")
    saveObj(valid_x, "valid_x")
    saveObj(test_x, "test_x")
    return

#______________________________ PART 2 ______________________________#
def part2():
    return

#______________________________ PART 3 ______________________________#
def part3():
    return

#______________________________ PART 4 ______________________________#
def part4():
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

    part1()
    #part2()
    #part3()
    #part4()
    #part5()
    #part6()
    #part7()
    #part8()

    end = time.time()
    print('Time elapsed:', end-start)
