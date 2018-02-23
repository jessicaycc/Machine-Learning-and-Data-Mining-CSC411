import torch
import torch.utils.data
import torch.nn as nn
from plot import *
from getdata import *
from torch.autograd import Variable

torch.manual_seed(0)

def initWeights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)
    return

def visWeights(m):
    if isinstance(m, nn.Linear):
        W = convert(np.array(m.weight.data), (-1,1), (0,255))
        for i in range(len(W)):
            heatmap(W[i], (32,32,3), 'pt9_weight_'+str(i))
    return

#______________________________ PART 8 ______________________________#
def part8():
    getData(act, (32, 32), download=False)

    dim_x = 3072
    dim_h = 30
    dim_out = 6

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    train_set, test_set, _ = getSets(act, (60, 20, 0), '32x32')
    train_x = convert(genX(train_set, dim_x, '32x32'), (0,1), (-1,1))
    train_y = convert(genY(train_set, dim_out),        (0,1), (-1,1))
    test_x  = convert(genX(test_set,  dim_x, '32x32'), (0,1), (-1,1))
    test_y  = convert(genY(test_set,  dim_out),        (0,1), (-1,1))
    
    batches = list()
    acc_test, acc_train = list(), list()
    loss_test, loss_train = list(), list()

    train_idx = np.random.permutation(range(train_x.shape[0]))
    x = torch.from_numpy(train_x[train_idx])
    y_classes = torch.from_numpy(np.argmax((train_y)[train_idx], 1))

    dataset = torch.utils.data.TensorDataset(x, y_classes)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    for data, target in train_loader:
        batches.append((
            Variable(data,   requires_grad=False).type(dtype_float),
            Variable(target, requires_grad=False).type(dtype_long)
        ))

    model = nn.Sequential(nn.Linear(dim_x, dim_h), nn.Tanh(), nn.Linear(dim_h, dim_out))
    model.apply(initWeights)

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    data_test    = Variable(torch.from_numpy(test_x),                requires_grad=False).type(dtype_float)
    target_test  = Variable(torch.from_numpy(np.argmax(test_y, 1)),  requires_grad=False).type(dtype_long)
    data_train   = Variable(torch.from_numpy(train_x),               requires_grad=False).type(dtype_float)
    target_train = Variable(torch.from_numpy(np.argmax(train_y, 1)), requires_grad=False).type(dtype_long)

    t = np.arange(1, 1001)
    for epoch in t:
        for data, target in batches:
            pred = model(data)
            loss = loss_fn(pred, target)
            model.zero_grad()
            loss.backward()  
            optimizer.step()
            
        pred = model(data_test)
        loss = loss_fn(pred, target_test)
        acc_test.append(np.mean(np.argmax(pred.data.numpy(), 1) == np.argmax(test_y, 1)))
        loss_test.append(loss.data.numpy())
        
        pred = model(data_train)
        loss = loss_fn(pred, target_train)
        acc_train.append(np.mean(np.argmax(pred.data.numpy(), 1) == np.argmax(train_y, 1)))
        loss_train.append(loss.data.numpy())
        
        if epoch == 700:
            saveObj(model, 'model')
        if epoch % 100 == 0:
            print('Epoch {} - completed'.format(epoch))

    print('Max accuracy:', max(acc_test))
    linegraphVec(acc_test, acc_train, t, 'pt8_learning_curve_accuracy')
    linegraphVec(loss_test, loss_train, t, 'pt8_learning_curve_loss')
    return

#______________________________ PART 9 ______________________________#
def part9():
    model = loadObj('model')
    model.apply(visWeights)
    return

#_______________________________ MAIN _______________________________#
if __name__ == '__main__':
    start = time.time()

    #part8()
    #part9()

    end = time.time()
    print('Time elapsed:', end-start)
