from const import *

class CNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_filters, kernel_size):
        super(CNN, self).__init__()

        self.classify = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size),
            nn.Conv2d(hidden_size, num_filters, kernel_size),
            nn.ReLU(),
            nn.Linear(num_filters, vocab_size),
            nn.Softmax(dim=1))

        self.apply(self.weight_init)
    
    def weight_init(self, x):
        if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear):
            nn.init.xavier_uniform(x.weights.data)
            nn.init.xavier_uniform(x.bias.data)

    def forward(self, x):
        self.out = self.classify(x)
        return self.out


def gen_vocab(data_file):
    vocab = list()
    for line in data_file:
        for word in line:
            if word not in vocab:
                vocab.append(word)
    vocab = sorted(vocab)
    return {k: v for v,k in enumerate(vocab)}

def gen_data_sets(data_file):
    with open(data_file) as f:
        shuffled = [l.split() for l in f]

    np.random.shuffle(shuffled)
    i, j, k = [int(s*len(shuffled)) for s in SET_RATIO]

    tra = shuffled[:i]
    val = shuffled[i:(i+j)]
    tes = shuffled[(i+j):(i+j+k)]
    return tra, val, tes

def word_to_num(data_list, vocab):
    numbers = [[vocab[w] for hl in data] for w in hl]
    padded = [hl + [0]*(MAX_HL_LEN-len(hl)) for hl in numbers]
    return np.asarray(padded)


def train(model, loss_fn, num_epochs, batch_size, learn_rate, reg_rate):
    train_x = torch.from_numpy(loadObj('train_x'))
    train_y = torch.from_numpy(loadObj('train_y'))
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learn_rate,
        weight_decay=reg_rate)
    
    model.train()

    train_acc = [test(model,'train')]
    valid_acc = [test(model,'valid')]

    for epoch in range(1, num_epochs+1):
        for i, (review, target) in enumerate(train_loader, 1):
            review = Variable(review, requires_grad=False).type(TF)
            target = Variable(target, requires_grad=False).type(TF)

            pred = model.forward(review).squeeze()
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch: [%d/%d], Steps: %d, Loss: %.4f' 
            % (epoch, num_epochs, len(train_dataset)//batch_size, loss.data[0]))

        train_acc.append(test(model,'train'))
        valid_acc.append(test(model,'valid'))

    linegraph(train_acc, valid_acc, np.arange(num_epochs+1), 'curve')
    return model

def test(model, set, th=0.5, batch_size=24):
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
        review = Variable(review, requires_grad=False).type(TF)
        target = Variable(target, requires_grad=False).type(TF)

        pred = model(review).squeeze().data.numpy()
        pred = (pred >= 0.5).astype(int)
        target = target.data.numpy()

        total += len(target)
        correct += np.sum(pred == target)
    
    model.train()

    return 100 * correct/total

def linegraph(y1, y2, x, filename):
    plt.plot(x, y1, label='training')
    plt.plot(x, y2, label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower left')
    plt.savefig('plots/'+filename+'.png', bbox_inches='tight')
    plt.show()
    return


if __name__ == '__main__':
    start = time.time()

    #TODO write main function

    end = time.time()
    print('Time elapsed: %.2fs' % (end-start))
