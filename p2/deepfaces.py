from plot import *
from getdata import *

class AlexNet(nn.Module):
    def init_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 9216)
        return x

class Classify(nn.Module):
    def init_weights(self):
        nn.init.xavier_uniform(self.classifier[0].weight.data)
        nn.init.constant(self.classifier[0].bias, 0.1)
        
        nn.init.xavier_uniform(self.classifier[2].weight.data)
        nn.init.constant(self.classifier[2].bias, 0.1)

    def __init__(self):
        super(Classify, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(9216, 30),
            nn.Tanh(),
            nn.Linear(30, 6)
        )
        
        self.init_weights()

    def forward(self, x):
        return self.classifier(x)

#______________________________ PART 10 _____________________________#
def part10():
    getData(act, (227, 227), download=False)
    imgs2obj(AlexNet().eval(), '227x227')

    dim_x = 9216
    dim_h = 30
    dim_out = 6

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    train_set, valid_set, test_set = getSets(act, (60, 20, 20), 'objects/AlexNet')
    train_x = genX(train_set, dim_x, 'objects/AlexNet')
    train_y = genY(train_set, dim_out)
    valid_x = genX(valid_set, dim_x, 'objects/AlexNet')
    valid_y = genY(valid_set, dim_out)
    test_x  = genX(test_set,  dim_x, 'objects/AlexNet')
    test_y  = genY(test_set,  dim_out)
    
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

    model = Classify().eval()
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)

    data_test    = Variable(torch.from_numpy(test_x),                requires_grad=False).type(dtype_float)
    target_test  = Variable(torch.from_numpy(np.argmax(test_y, 1)),  requires_grad=False).type(dtype_long)
    data_train   = Variable(torch.from_numpy(train_x),               requires_grad=False).type(dtype_float)
    target_train = Variable(torch.from_numpy(np.argmax(train_y, 1)), requires_grad=False).type(dtype_long)
    data_valid    = Variable(torch.from_numpy(valid_x),                requires_grad=False).type(dtype_float)
    target_valid  = Variable(torch.from_numpy(np.argmax(valid_y, 1)),  requires_grad=False).type(dtype_long)

    t = np.arange(1, 1001)
    for epoch in t:
        for data, target in batches:
            pred = model.forward(data)
            loss = loss_fn(pred, target)
            model.classifier.zero_grad()
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
        
        if epoch % 100 == 0:
            print('Epoch {} - completed'.format(epoch))

    print('Max accuracy:', max(acc_test))
    linegraphVec(acc_valid, acc_train, t, 'pt10_learning_curve_accuracy')
    linegraphVec(loss_valid, loss_train, t, 'pt10_learning_curve_loss')
    return

#_______________________________ MAIN _______________________________#
if __name__ == '__main__':
    start = time.time()

    part10()

    end = time.time()
    print('Time elapsed:', end-start)
