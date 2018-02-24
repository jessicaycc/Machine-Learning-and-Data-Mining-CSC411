from plot import *
from getdata import *

#______________________________ PART 8 ______________________________#
def part8():
    #getData(act, (32, 32), download=False)

    dim_x = 3072
    dim_h = 30
    dim_out = 6

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    train_set, valid_set, test_set = getSets(act, (40, 20, 20), 'processed/32x32')
    train_x = convert(genX(train_set, dim_x, 'processed/32x32'), (0,1), (-1,1))
    valid_x  = convert(genX(valid_set,  dim_x, 'processed/32x32'), (0,1), (-1,1))
    test_x = convert(genX(test_set,  dim_x, 'processed/32x32'), (0,1), (-1,1))
    train_y = convert(genY(train_set, dim_out), (0,1), (-1,1))
    valid_y  = convert(genY(valid_set,  dim_out), (0,1), (-1,1))
    test_y = convert(genY(test_set,  dim_out), (0,1), (-1,1))
    
    batches = list()
    acc_valid, acc_train, acc_test = list(), list(), list()
    loss_valid, loss_train, loss_test = list(), list(), list()

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
    nn.init.xavier_uniform(model[0].weight.data)
    nn.init.constant(model[0].bias, 0.1)
    nn.init.xavier_uniform(model[2].weight.data)
    nn.init.constant(model[2].bias, 0.1)

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    data_valid    = Variable(torch.from_numpy(valid_x),                requires_grad=False).type(dtype_float)
    target_valid  = Variable(torch.from_numpy(np.argmax(valid_y, 1)),  requires_grad=False).type(dtype_long)
    data_train   = Variable(torch.from_numpy(train_x),               requires_grad=False).type(dtype_float)
    target_train = Variable(torch.from_numpy(np.argmax(train_y, 1)), requires_grad=False).type(dtype_long)
    data_test   = Variable(torch.from_numpy(test_x),               requires_grad=False).type(dtype_float)
    target_test = Variable(torch.from_numpy(np.argmax(test_y, 1)), requires_grad=False).type(dtype_long)
    t = np.arange(1, 1001)
    for epoch in t:
        for data, target in batches:
            pred = model(data)
            loss = loss_fn(pred, target)
            model.zero_grad()
            loss.backward()  
            optimizer.step()
            
        pred = model(data_valid)
        loss = loss_fn(pred, target_valid)
        acc_valid.append(np.mean(np.argmax(pred.data.numpy(), 1) == np.argmax(valid_y, 1)))
        loss_valid.append(loss.data.numpy())
        
        pred = model(data_train)
        loss = loss_fn(pred, target_train)
        acc_train.append(np.mean(np.argmax(pred.data.numpy(), 1) == np.argmax(train_y, 1)))
        loss_train.append(loss.data.numpy())
        
        pred = model(data_test)
        loss = loss_fn(pred, target_test)
        acc_test.append(np.mean(np.argmax(pred.data.numpy(), 1) == np.argmax(test_y, 1)))
        loss_test.append(loss.data.numpy())

        if epoch == 700:
            saveObj(model, 'model')
        if epoch % 100 == 0:
            print('Epoch {} - completed'.format(epoch))

    print('Max accuracy:', max(acc_test))
    linegraphVec(acc_valid, acc_train, t, 'pt8_learning_curve_accuracy')
    linegraphVec(loss_valid, loss_train, t, 'pt8_learning_curve_loss')
    return

#______________________________ PART 9 ______________________________#
def part9():
    dtype_float = torch.FloatTensor

    train_set, _, _ = getSets(act, (60, 20, 0), 'processed/32x32')
    train_x = genX(train_set, 3072, 'processed/32x32')

    model = loadObj('model')
    visWeights(model[0])
    model = nn.Sequential(model[0], model[1])   

    img = train_x[:84,:]
    X = Variable(torch.from_numpy(img), requires_grad = False).type(dtype_float)
    out = model.forward(X).data.numpy()
    activation = np.argmax(out, 1).tolist()
    print('Lorraine Bracco: {}'.format(max(set(activation), key=activation.count)))

    img = train_x[449:556,:]
    X = Variable(torch.from_numpy(img), requires_grad = False).type(dtype_float)
    out = model.forward(X).data.numpy()
    activation = np.argmax(out, 1).tolist()
    print('Steven Carell: {}'.format(max(set(activation), key=activation.count)))

    return

#_______________________________ MAIN _______________________________#
if __name__ == '__main__':
    start = time.time()

    #part8()
    part9()

    end = time.time()
    print('Time elapsed:', end-start)
