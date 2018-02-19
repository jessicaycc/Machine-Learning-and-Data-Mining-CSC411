import os
import hashlib
import threading
from const import *
from PIL import Image
#from urllib import urlretrieve
from urllib.request import urlretrieve

if not os.path.exists("original"):
    os.makedirs("original")

if not os.path.exists("processed"):
    os.makedirs("processed")

def timeout(func, args=(), kwargs={}, timeout_duration=10, default=None):
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
        return(False)
    else:
        return(it.result)

def process(filename, data):
    bounding_box = tuple(map(int, data[DATA_BBOX].split(",")))
    with open("original/"+filename, "rb") as infile:
        hash_matches = data[DATA_HASH] == hashlib.sha256(infile.read()).hexdigest()
    img = Image.open("original/"+filename).crop(bounding_box).resize(DATA_SIZE)
    is_coloured = img.mode not in ("L", "LA")
    if hash_matches and is_coloured:
        img.save("processed/" + filename)
    else:
        err_msg = "hash does not match" if not hash_matches else "image has no colour"
        raise IOError(err_msg)
    return

def getData(act, download=True):
    def search(name, textFile, download):
        i = 0
        for line in open(textFile):
            if name in line:
                data = line.strip("\r\n").split("\t")
                ext = "." + data[DATA_URL].split(".")[-1]
                filename = data[DATA_NAME].replace(" ","_").lower() + str(i) + ext
                if download:
                    timeout(urlretrieve, (data[DATA_URL], "original/"+filename), {})
                if os.path.isfile("original/"+filename):
                    try:
                        process(filename, data)
                        print(filename, "- success")
                    except IOError as err:
                        print("{} - failed with error: {}".format(filename, err.args[0]))
                else:
                    print(filename, "- failed with error: image failed to download")
                i += 1
        return
    for name in act:
        search(name, "data/facescrub_actors.txt", download)
        search(name, "data/facescrub_actresses.txt", download)
    return

def getSets(act, set_ratio=DATA_SET_RATIO):
    set_ratio = [float(x)/sum(set_ratio) for x in set_ratio]
    set1, set2, set3 = list(), list(), list()
    for name in act:
        filename = name.replace(" ","_").lower()
        database = [f for f in os.listdir("processed") if f.startswith(filename)]
        size = [int(x*len(database)) for x in set_ratio]
        sample = np.random.choice(database, sum(size), replace=False)
        set1.append(sample[:size[0]])
        set2.append(sample[size[0]:size[0]+size[1]])
        set3.append(sample[size[0]+size[1]:])
    return(set1, set2, set3)

def genX(file_set):
    X = np.empty((0, VEC_SIZE), float)
    for file_list in file_set:
        for filename in file_list:
            img = Image.open("processed/"+filename)
            x = np.array(img)[:,:,:3].flatten() / 255.
            X = np.vstack((X, x))
    return(X)

def genY(file_set):
    labels = np.identity(NUM_ACT)
    Y = np.empty((0, NUM_ACT), float)
    for i, file_list in enumerate(file_set):
        size = len(file_list)
        Y = np.vstack(( Y, np.tile(labels[i], (size, 1)) ))
    return(Y)
