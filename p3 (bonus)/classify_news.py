#!/usr/bin/python3
from fakebonus import *

def gen_data(data_file):
    with open(data_file) as f:
        data = [l.split() for l in f]
    return data

model = loadObj('model')
vocab = loadObj('vocab')

test_x = gen_data(sys.argv[1])
test_x = one_hot(test_x, vocab)
test_x = torch.from_numpy(test_x)

model.eval()
for hl in test_x:
    hl = Variable(hl, requires_grad=False).type(TF)
    pred = model(hl).squeeze().data.numpy()
    pred = (pred >= 0.5).astype(int)
    print(pred[0])
