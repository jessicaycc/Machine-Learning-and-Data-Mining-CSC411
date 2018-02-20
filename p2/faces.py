import torch
import torch.utils.data
import torch.nn as nn
from const import *
from plot import *
from getdata import *
from torch.autograd import Variable

torch.manual_seed(0)
act = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon", "Alec Baldwin", "Bill Hader", "Steve Carell"]

def initWeights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform(m.weight.data)
        nn.init.constant(m.bias, 0.1)
    return

def visWeights(m):
    if isinstance(m, nn.Linear):
        W = convert(np.array(m.weight.data), (-1,1), (0,255))
        for i in range(len(W)):
            heatmap(W[i], (32,32,3), "pt9_weight_"+str(i))
    return

#______________________________ PART 8 ______________________________#
def part8():
    #getData(act, download=False)
    train_set, valid_set, test_set= getSets(act)
    train_x = genX(train_set)
    train_y = genY(train_set)
    test_x = genX(test_set)
    test_y = genY(test_set)

    dim_x = 3072
    dim_h = 30
    dim_out = 6

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor
    
    t = np.arange(1, 1001)
    acc_test, acc_train, batches = [], [], []

    train_idx = np.random.permutation(range(train_x.shape[0]))
    x = torch.from_numpy(train_x[train_idx])
    y_classes = torch.from_numpy(np.argmax((train_y)[train_idx], 1))

    dataset = torch.utils.data.TensorDataset(x, y_classes)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    for data, target in train_loader:
        batches.append((
            Variable(data, requires_grad=False).type(dtype_float),
            Variable(target, requires_grad=False).type(dtype_long)
        ))

    model = nn.Sequential(nn.Linear(dim_x, dim_h), nn.Tanh(), nn.Linear(dim_h, dim_out))
    model.apply(initWeights)

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
    train_x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)

    for epoch in t:
        for data, target in batches:
            pred = model(data)
            loss = loss_fn(pred, target)
            model.zero_grad()
            loss.backward()  
            optimizer.step()
            
        y_pred = model(test_x).data.numpy()
        acc_test.append( np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1)) )
        y_pred = model(train_x).data.numpy()
        acc_train.append( np.mean(np.argmax(y_pred, 1) == np.argmax(train_y, 1)) )
        
        if epoch == 700:
            saveObj(model, "model")
        if epoch % 100 == 0:
            print("Epoch {} - completed".format(epoch))

    print("Max accuracy:", max(acc_test))
    linegraphVec(acc_test, acc_train, t, "pt8_learning_curve")
    return

#______________________________ PART 9 ______________________________#
def part9():
    model = loadObj("model")
    model.apply(visWeights)
    return

#_______________________________ MAIN _______________________________#
if __name__ == "__main__":
    start = time.time()

    #part8()
    #part9()

    end = time.time()
    print("Time elapsed:", end-start)
