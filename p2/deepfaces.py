import torch
import torch.utils.data
from const import *
from plot import *
from getdata import *
from torch.autograd import Variable

torch.manual_seed(0)
act = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon", "Alec Baldwin", "Bill Hader", "Steve Carell"]

def initWeights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.constant(m.bias, 0.1)
    return

def visWeights(m):
    if isinstance(m, torch.nn.Linear):
        W = convert(np.array(m.weight.data), (-1,1), (0,1))
        for i in range(len(W)):
            heatmap(W[i], "pt9_weight_"+str(i), (32,32,3))
    return

#______________________________ PART 8 ______________________________#
def part8():
    #getData(act, download=False)
    train_set, valid_set, test_set= getSets(act)
    train_x = genX(train_set)
    train_y = genY(train_set)
    test_x = genX(test_set)
    test_y = genY(test_set)

    dim_x = IN_SIZE
    dim_h = 30
    dim_out = OUT_SIZE

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    train_idx = np.random.permutation(range(train_x.shape[0]))[:550]
    x = torch.from_numpy(train_x[train_idx])
    y_classes = torch.from_numpy(np.argmax((train_y)[train_idx], 1))

    model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h), torch.nn.Tanh(), torch.nn.Linear(dim_h, dim_out))
    model.apply(initWeights)

    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = torch.utils.data.TensorDataset(x, y_classes)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=50)

    for epoch, (data, target) in enumerate(train_loader):
        print("Epoch", epoch+1)
        data = Variable(data, requires_grad=False).type(dtype_float)
        target = Variable(target, requires_grad=False).type(dtype_long)
        for t in range(2000):
            pred = model(data)
            loss = loss_fn(pred, target)
            model.zero_grad()
            loss.backward()  
            optimizer.step()
            if (t+1) % 100 == 0:
                print("Iter", t+1)
        print("\n")

    #saveObj(model, "model")

    x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()
    accuracy = np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))
    print(accuracy)
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
    part9()

    end = time.time()
    print("Time elapsed:", end-start)