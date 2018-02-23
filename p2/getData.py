import hashlib
import threading
from const import *
from PIL import Image
from urllib.request import urlretrieve

if not os.path.exists('original'):
    os.makedirs('original')

if not os.path.exists('processed'):
    os.makedirs('processed')

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

def process(filename, data, res):
    bounding_box = tuple(map(int, data[DATA_BBOX].split(',')))

    with open('original/' + filename, 'rb') as infile:
        hash_matches = data[DATA_HASH] == hashlib.sha256(infile.read()).hexdigest()
    img = Image.open('original/' + filename).crop(bounding_box).resize(res)
    is_coloured = img.mode not in ('L', 'LA')

    if hash_matches and is_coloured:
        dir = '{}x{}'.format(res[0], res[1])
        if not os.path.exists('processed/'+dir):
            os.makedirs('processed/'+dir)
        img.save('processed/{}/{}'.format(dir, filename))
    else:
        err_msg = 'hash does not match' if not hash_matches else 'image has no colour'
        raise IOError(err_msg)

    return

def search(name, textFile, res, download):
    for i, line in enumerate(l for l in open(textFile) if name in l):
        data = line.strip('\r\n').split('\t')
        ext = '.' + data[DATA_URL].split('.')[-1]
        filename = data[DATA_NAME].replace(' ', '_').lower() + str(i) + ext

        if download:
            timeout(urlretrieve, (data[DATA_URL], 'original/'+filename), {})

        if os.path.isfile('original/'+filename):
            try:
                process(filename, data, res)
                print(filename, '- success')
            except IOError as err:
                print('{} - failed with error: {}'.format(filename, err.args[0]))
        else:
            print(filename, '- failed with error: image failed to download')

    return

def getData(act, res, download=True):
    for name in act:
        search(name, 'data/facescrub_actors.txt', res, download)
        search(name, 'data/facescrub_actresses.txt', res, download)
    return

def getSets(act, set_ratio, dir):
    set1, set2, set3 = list(), list(), list()
    set_ratio = [float(x)/sum(set_ratio) for x in set_ratio]
    for name in act:
        filename = name.replace(' ', '_').lower()
        database = [f for f in os.listdir('processed/'+dir) if f.startswith(filename)]
        size = [int(x*len(database)) for x in set_ratio]
        sample = np.random.choice(database, sum(size), replace=False)
        set1.append(sample[:size[0]])
        set2.append(sample[size[0]:size[0]+size[1]])
        set3.append(sample[size[0]+size[1]:])
    return set1, set2, set3

def genX(file_set, num_features, dir):
    X = np.empty((0, num_features), float)
    for file_list in file_set:
        for filename in file_list:
            img = Image.open('processed/{}/{}'.format(dir, filename))
            x = np.array(img)[:,:,:3].flatten() / 255.
            X = np.vstack((X, x))
    return X

def genY(file_set, num_labels):
    labels = np.identity(num_labels)
    Y = np.empty((0, num_labels), float)
    for i, file_list in enumerate(file_set):
        size = len(file_list)
        Y = np.vstack(( Y, np.tile(labels[i], (size, 1)) ))
    return Y

def convert(A, range_old, range_new):
    len_old = range_old[1]-range_old[0]
    len_new = range_new[1]-range_new[0]
    old_min = np.ones(A.shape)*range_old[0]
    new_min = np.ones(A.shape)*range_new[0]
    return (A - old_min)*len_new/len_old + new_min
