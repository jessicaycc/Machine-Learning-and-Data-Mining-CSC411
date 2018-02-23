import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.models as models
from plot import *
from getdata import *
from torch.autograd import Variable
from scipy.misc import imread, imresize

torch.manual_seed(0)

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
        x = x.view(x.size(0), 256*6*6)
        return x

class Classify(nn.Module):
    def init_weights(self):
        nn.init.xavier_uniform(self.classifier[0].weight.data)
        nn.init.constant(self.classifier[0].bias, 0.1)
        
        nn.init.xavier_uniform(self.classifier[2].weight.data)
        nn.init.constant(self.classifier[2].bias, 0.1)

    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256*6*6, 30),
            nn.Tanh(inplace=True),
            nn.Linear(30, 6)
        )
        
        self.init_weights()

    def forward(self, x):
        return self.classifier(x)

def img2obj(model, filename, dir):
    im = imread('processed/{}/{}'.format(dir, filename))[:,:,:3]
    im = im - np.mean(im.flatten())
    im = im/np.max(np.abs(im.flatten()))
    im = np.rollaxis(im, -1).astype(np.float32)

    im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)
    im_v = model.forward(im_v)
    filename = filename.split('.')[0]

    if not os.path.exists('objects/AlexNet'):
        os.makedirs('objects/AlexNet')
    try:
        saveObj(im_v, 'AlexNet/' + filename)
        print(filename, '- success')
    except IOError as err:
        print('{} - failed with error: {}'.format(filename, err.args[0]))
    return

#______________________________ PART 10 _____________________________#
def part10():
    #getData(act, (227, 227), download=False)

    model = AlexNet()
    model.eval()

    for f in os.listdir('processed/227x227'):
        img2obj(model, f, '227x227')

#    softmax = nn.Softmax()
#    all_probs = softmax(model.forward(im_v)).data.numpy()[0]
#    sorted_ans = np.argsort(all_probs)
#
#    for i in range(-1, -6, -1):
#        print('Answer:', class_names[sorted_ans[i]], ', Prob:', all_probs[sorted_ans[i]])
#
#    ans = np.argmax(model.forward(im_v).data.numpy())
#    prob_ans = softmax(model.forward(im_v)).data.numpy()[0][ans]
#    print('Top Answer:', class_names[ans], 'P(ans) = ', prob_ans)
    return

#_______________________________ MAIN _______________________________#
if __name__ == '__main__':
    start = time.time()

    part10()

    end = time.time()
    print('Time elapsed:', end-start)
