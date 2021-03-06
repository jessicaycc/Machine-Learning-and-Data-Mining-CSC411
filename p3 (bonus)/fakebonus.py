from const import *

class CNN(nn.Module):
    def __init__(self, vocab_size):
        super(CNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, 128)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1920, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.embed(x).unsqueeze(1)
        x = self.conv(x)
        x = self.fc(x.view(x.size(0), 1920))
        return x

class LogisticRegression(nn.Module):
    def init_weights(self):
        nn.init.xavier_uniform(self.features[1].weight.data)
        nn.init.constant(self.features[1].bias, 0.1)

    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.features = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size, 1),
            nn.Sigmoid())
        
        self.init_weights()

    def forward(self, x):
        return self.features(x)


def gen_vocab(data_file):
    vocab = list()
    for ln in data_file:
        for w in ln:
            if w not in vocab:
                vocab.append(w)
    vocab = [PAD_WORD] + sorted(vocab)
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

def gen_data_labels(data_list, test_p3=False):
    size = {
        'tra': SET_RATIO[0],
        'val': SET_RATIO[1],
        'tes': SET_RATIO[2]
        }[data_list]
    if test_p3:
        real = np.zeros(int(size*NUM_REAL_P3))
        fake = np.ones(int(size*NUM_FAKE_P3))
    else:
        real = np.zeros(int(size*NUM_REAL))
        fake = np.ones(int(size*NUM_FAKE))
    return np.concatenate((real, fake))

def word_to_num(data_list, vocab):
    numbers = np.zeros((len(data_list), MAX_HL_LEN))
    for i, hl in enumerate(data_list):
        for j, w in enumerate(hl):
            if j >= MAX_HL_LEN:
                break
            if w in vocab:
                numbers[i][j] = vocab[w]
    return numbers

def one_hot(data_list, vocab):
    X = np.zeros((len(data_list), len(vocab)))
    for i, line in enumerate(data_list):
        for word in line:
            if word in vocab:
                X[i][vocab[word]] = 1.
    return X


def train(model, loss_fn, num_epochs, batch_size, learn_rate, reg_rate):
    train_x = torch.from_numpy(loadObj('tra_x'))
    train_y = torch.from_numpy(loadObj('tra_y'))
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learn_rate,
        weight_decay=reg_rate)
    
    tra_acc = [test(model,'tra')]
    val_acc = [test(model,'val')]
    max_acc = val_acc[0]

    model.train()
    saveObj(model, 'model')

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

        tra_acc.append(test(model,'tra'))
        val_acc.append(test(model,'val'))

        if val_acc[-1] > max_acc:
            saveObj(model, 'model')

    learn_curve(tra_acc, val_acc, np.arange(num_epochs+1))
    return

def test(model, data_set, test_p3=False):
    if test_p3:
        test_x = torch.from_numpy(loadObj('tes_x_p3'))
        test_y = torch.from_numpy(loadObj('tes_y_p3'))
    else:
        test_x = torch.from_numpy(loadObj(data_set+'_x'))
        test_y = torch.from_numpy(loadObj(data_set+'_y'))
        
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=128, 
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

def learn_curve(y1, y2, x):
    plt.plot(x, y1, label='training')
    plt.plot(x, y2, label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower left')
    plt.savefig('plots/learn_curve.png', bbox_inches='tight')
    return

def disp_keywords():
    vocab = loadObj('vocab')
    model = loadObj('model')

    vocab = list(vocab.keys())
    W = model.features[1].weight.data.numpy()[0]
    W_index_sorted = W.argsort()

    W_pos = W_index_sorted[-10:][::-1]
    W_neg = W_index_sorted[:10]

    top10_pos = [(W[i], vocab[i]) for i in W_pos]
    top10_neg = [(W[i], vocab[i]) for i in W_neg]

    print('Top 10 positive weights:', top10_pos)
    print('\nTop 10 negative weights:', top10_neg)
    
    W_pos = W_index_sorted[::-1]
    W_neg = W_index_sorted[:]

    top10_pos = [(W[i], vocab[i]) for i in W_pos if vocab[i] not in ENGLISH_STOP_WORDS][:10]
    top10_neg = [(W[i], vocab[i]) for i in W_neg if vocab[i] not in ENGLISH_STOP_WORDS][:10]

    print('\nTop 10 positive weights (no stop words):', top10_pos)
    print('\nTop 10 negative weights (no stop words):', top10_neg)
    return


def init_data():
    tra, val, tes = (a+b for a,b in zip(gen_data_sets('clean_real.txt'), gen_data_sets('clean_fake.txt')))

    vocab = gen_vocab(tra)
    tra_x = one_hot(tra, vocab)
    val_x = one_hot(val, vocab)
    tes_x = one_hot(tes, vocab)
    tra_y = gen_data_labels('tra')
    val_y = gen_data_labels('val')
    tes_y = gen_data_labels('tes')

    saveObj(vocab, 'vocab')
    saveObj(tra_x, 'tra_x')
    saveObj(val_x, 'val_x')
    saveObj(tes_x, 'tes_x')
    saveObj(tra_y, 'tra_y')
    saveObj(val_y, 'val_y')
    saveObj(tes_y, 'tes_y')

    _, _, tes = (a+b for a,b in zip(gen_data_sets('clean_real_p3.txt'), gen_data_sets('clean_fake_p3.txt')))

    tes_x = one_hot(tes, vocab)
    tes_y = gen_data_labels('tes', test_p3=True)

    saveObj(tes_x, 'tes_x_p3')
    saveObj(tes_y, 'tes_y_p3')
    return


if __name__ == '__main__':
    start = time.time()

    init_data()
    VOCAB_SIZE = len(loadObj('vocab'))

    train(model=LogisticRegression(VOCAB_SIZE),
          loss_fn=nn.BCELoss(),
          num_epochs=100,
          batch_size=128,
          learn_rate=1e-3,
          reg_rate=0)

    model = loadObj('model')

    print('Accuracy on train set: %.2f%%' % test(model,'tra'))
    print('Accuracy on valid set: %.2f%%' % test(model,'val'))
    print('Accuracy on test set: %.2f%%'  % test(model,'tes'))
    print('Accuracy on test set (p3): %.2f%%' % test(model, '', test_p3=True))
    print('\n')

    disp_keywords()

    end = time.time()
    print('Time elapsed: %.2fs' % (end-start))
