from numpy import dot

def softmax(y):
    return exp(y)/tile(sum(exp(y),0), (len(y),1))

def forward(x, W, b)
    return softmax( dot(W.T, x) + b )