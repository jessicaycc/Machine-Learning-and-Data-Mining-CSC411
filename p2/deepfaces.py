import torch
from const import *
from getdata import *
from torch.autograd import Variable

act = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon", "Alec Baldwin", "Bill Hader", "Steve Carell"]

#getData(act, download=False)
train_set, valid_set, test_set= getSets(act)
train_x = genX(train_set)
train_y = genY(train_set)
test_x = genX(test_set)
test_y = genY(test_set)

dim_x = 32*32*3
dim_h = 30
dim_out = 6

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

train_idx = np.random.permutation(range(train_x.shape[0]))[:360]
x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
y_classes = Variable(torch.from_numpy(np.argmax((train_y)[train_idx], 1)), requires_grad=False).type(dtype_long)

model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h), torch.nn.Sigmoid(), torch.nn.Linear(dim_h, dim_out))

loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(10000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)
    model.zero_grad()
    loss.backward()  
    optimizer.step()

x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
y_pred = model(x).data.numpy()
accuracy = np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))
print(accuracy)
