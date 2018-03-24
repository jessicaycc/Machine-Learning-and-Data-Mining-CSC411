EMBEDDING_SIZE = 10
embeds = tf.contrib.layers.embed_sequence(sliced, 
                 vocab_size=N_WORDS, embed_dim=EMBEDDING_SIZE)

WINDOW_SIZE = EMBEDDING_SIZE
STRIDE = int(WINDOW_SIZE/2)
conv = tf.contrib.layers.conv2d(embeds, 1, WINDOW_SIZE, stride=STRIDE, padding='SAME') # (?, 4, 1)    
conv = tf.nn.relu(conv) # (?, 4, 1)    
words = tf.squeeze(conv, [2]) # (?, 4)

n_classes = len(TARGETS)     
logits = tf.contrib.layers.fully_connected(words, n_classes, activation_fn=None)

predictions_dict = {      
    'source': tf.gather(TARGETS, tf.argmax(logits, 1)),
    'class': tf.argmax(logits, 1),
    'prob': tf.nn.softmax(logits)
}

class CNN(nn.Module):
    def __init__(self, vocab_size, input_size, kernel_size):
        super(CNN, self).__init__()

        self.classify = nn.Sequential(
            nn.Embedding(vocab_size, input_size),
            nn.Conv1d(input_size, 1, kernel_size),
            nn.ReLU(),
            nn.Linear(18, 1),
            nn.Sigmoid())

        # self.apply(self.weight_init)

    def weight_init(self, x):
        if isinstance(x, nn.Conv2d) or isinstance(x, nn.Linear):
            nn.init.xavier_uniform(x.weight.data)
            nn.init.xavier_uniform(x.bias.data)

    def forward(self, x):
        return self.classify(x)