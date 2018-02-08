import urllib
import threading
from const import *

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
    def process(filename, data):
        boundingBox = tuple(map(int, data[DATA_BBOX].split(",")))
        img = Image.open("original/" + filename)
        img = img.crop(boundingBox).resize(DATA_SIZE).convert("L")
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
                        process(filename, data)
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
