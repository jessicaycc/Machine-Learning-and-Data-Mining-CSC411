from plot import *
from const import *

class LogisticRegression(nn.Module):
    def init_weights(self):
        nn.init.xavier_uniform(self.features[1].weight.data)
        nn.init.constant(self.features[1].bias, 0.1)

    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.features = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size, num_classes),
            nn.Sigmoid())
        
        self.init_weights()

    def forward(self, x):
        return self.features(x)

def L1(model):
    reg = Variable(torch.FloatTensor(1), requires_grad=True).type(dtype_float)
    for W in model.parameters():
        reg = reg + W.norm(1)
    return reg

def L2(model):
    reg = Variable(torch.FloatTensor(1), requires_grad=True).type(dtype_float)
    for W in model.parameters():
        reg = reg + W.norm(2)
    return reg

def train(model, loss_fn, num_epochs, batch_size, learn_rate, reg_rate):
    train_x = torch.from_numpy(loadObj('train_x'))
    train_y = torch.from_numpy(loadObj('train_y'))
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True)

    train_acc, valid_acc = list(), list()
    epochs = np.arange(num_epochs + 1)
    num_steps = len(train_dataset) // batch_size

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learn_rate,
        weight_decay=reg_rate)

    train_acc.append(test(model,'train'))
    valid_acc.append(test(model,'valid'))

    model.train()

    for epoch in range(num_epochs):
        for i, (review, target) in enumerate(train_loader, 1):
            review = Variable(review, requires_grad=False).type(dtype_float)
            target = Variable(target, requires_grad=False).type(dtype_long)

            pred = model.forward(review)
            loss = loss_fn(pred, target)
            model.features.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch: [%d/%d], Steps: %d, Loss: %.4f' 
            % (epoch+1, num_epochs, num_steps, loss.data[0]))

        train_acc.append(test(model,'train'))
        valid_acc.append(test(model,'valid'))

    linegraph(train_acc, valid_acc, epochs, 'pt4_curve')
    return model

def test(model, set, batch_size=10):
    test_x = torch.from_numpy(loadObj(set+'_x'))
    test_y = torch.from_numpy(loadObj(set+'_y'))
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False)
    
    model.eval()
    
    correct, total = 0, 0
    for review, target in test_loader:
        review = Variable(review, requires_grad=False).type(dtype_float)
        pred = model(review)
        _, pred = torch.max(pred.data, 1)
        total += target.size(0)
        correct += (pred == target.type(dtype_long)).sum()

    return 100 * correct/total
