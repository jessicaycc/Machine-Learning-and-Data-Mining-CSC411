import torch
import torch.utils.data
from const import *
from getdata import *
from torch.autograd import Variable

torch.manual_seed(0)
act = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon", "Alec Baldwin", "Bill Hader", "Steve Carell"]

def initWeights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data)
        torch.nn.init.constant(m.bias, 0.1)

#getData(act, download=False)
train_set, valid_set, test_set= getSets(act)
train_x = genX(train_set)
train_y = genY(train_set)
test_x = genX(test_set)
test_y = genY(test_set)

train_x = convert(train_x, (0,1), (-1,1))
train_y = convert(train_y, (0,1), (-1,1))
test_x = convert(test_x, (0,1), (-1,1))
test_y = convert(test_y, (0,1), (-1,1))

dim_x = 32*32*3
dim_h = 30
dim_out = 6

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
    for t in range(3000):
        pred = model(data)
        loss = loss_fn(pred, target)
        model.zero_grad()
        loss.backward()  
        optimizer.step()
        if (t+1) % 100 == 0:
            print("Iter", t+1)
    print("\n")

x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
y_pred = model(x).data.numpy()
accuracy = np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))
print(accuracy)
