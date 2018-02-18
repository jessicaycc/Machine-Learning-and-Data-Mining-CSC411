import urllib
import threading
import hashlib
from hashlib import sha256
from part8Const import *

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except IOError:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def getData(act, download=True):
    def process(filename, data, line):
        boundingBox = tuple(map(int, data[DATA_BBOX].split(",")))
        img = Image.open("original/" + filename)
        img = img.crop(boundingBox).resize(DATA_SIZE)
        h = hashlib.sha256(open("original/" + filename, "rb").read()).hexdigest()
        print h
        if line.split()[6] == h:
            img.save("processed/" + filename)
        return

    def search(name, textFile, download):
        i = 0
        for line in open(textFile):
            if name in line:
                data = line.strip("\r\n").split("\t")
                ext = "." + data[DATA_URL].split(".")[-1]
                filename = data[DATA_NAME].replace(" ","_").lower() + str(i) + ext
                if download:
                    timeout(urllib.urlretrieve, (data[DATA_URL], "original/" + filename), {}, 10)
                if os.path.isfile("original/" + filename):
                    try:
                        process(filename, data, line)
                        print filename, "- success"
                    except IOError:
                        print filename, "- failed"
                else:
                    print filename, "- failed"
                i += 1
        return

    if not os.path.exists("original"):
        os.makedirs("original")
    if not os.path.exists("processed"):
        os.makedirs("processed")
    for name in act:
        search(name, "data/facescrub_actors.txt", download)
        search(name, "data/facescrub_actresses.txt", download)
    return

def getSets(act, rand=True, setRatio=DATA_SET_RATIO):
    sampleSize = sum(setRatio)
    set1, set2, set3 = list(), list(), list()
    for name in act:
        filename = name.replace(" ","_").lower()
        database = [f for f in os.listdir("processed") if f.startswith(filename)]
        if sampleSize > len(database):
            print "ERROR from getSets() - sample size is greater than size of database"
            quit()
        if rand:
            sample = np.random.choice(database, sampleSize, replace=False)
        else:
            sample = database[:sampleSize]
        set1.append(sample[:setRatio[0]])
        set2.append(sample[setRatio[0]:setRatio[0]+setRatio[1]])
        set3.append(sample[setRatio[0]+setRatio[1]:])
    return set1, set2, set3

def getTrain(trainSet):
    N = len(trainSet[0])
    I = np.identity(6)
    trainSet_X = trainSet
    trainSet_Y = np.empty((6, 0), float)
    for i in range(0, 6):
        trainSet_Y = np.hstack( (trainSet_Y, np.tile(I[:,[i]], (1, N))) )
    return trainSet_X, trainSet_Y

def getTest(testSet):
    N = len(testSet[0])
    I = np.identity(6)
    testSet_X = testSet 
    testSet_Y = np.empty((6, 0), float)
    for i in range(0, 6):
        testSet_Y = np.hstack( (testSet_Y, np.tile(I[:,[i]], (1, N))) )
    return testSet_X, testSet_Y

def genMatrix(fileList):
    M = np.empty((0, VEC_SIZE), float)
    for filename in fileList:
        img = Image.open("processed/" + filename)
        print filename
        print np.shape(img)
        x = np.array(img).flatten() / 255.
        print "x", np.shape(x)
        M = np.vstack((M, x))
    return np.append(M, np.ones((len(M), 1)), axis=1)