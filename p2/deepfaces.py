import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from getData import *

act = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon", "Alec Baldwin", "Bill Hader", "Steve Carell"]

#getData(act)
trainSet, validSet, testSet= getSets(act)
trainSet = genMatrix(np.asarray(trainSet).flatten()).T
testSet = genMatrix(np.asarray(testSet).flatten()).T
train_x, train_y = getTrain(trainSet)
test_x, test_y = getTest(testSet)

dim_x = 32*32
dim_h = 20
dim_out = 6

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

train_idx = np.random.permutation(range(train_x.shape[0]))[:480]
x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
y_classes = Variable(torch.from_numpy(np.argmax((train_y.T)[train_idx], 1)), requires_grad=False).type(dtype_long)

model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h), torch.nn.ReLU(), torch.nn.Linear(dim_h, dim_out))

loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-2
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
print accuracy