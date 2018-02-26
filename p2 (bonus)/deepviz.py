import time
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from plot import *
from scipy.misc import imread, imresize

class AlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)

        features_weight_i = [0, 3, 6, 8, 10]
        conv_layers = [a for a in dir(self) if a.startswith('conv')]
        for i, layer in zip(features_weight_i, conv_layers):
            layer = getattr(self, layer)
            layer[0].weight = an_builtin.features[i].weight
            layer[0].bias = an_builtin.features[i].bias
            
        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(9216, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes))
        
        self.load_weights()

    def forward(self, x):
        self.out1 = self.conv1(x)
        self.out2 = self.conv2(self.out1)
        self.out3 = self.conv3(self.out2)
        self.out4 = self.conv4(self.out3)
        self.out5 = self.conv5(self.out4)

        self.out6 = self.out5.view(self.out5.size(0), 9216)
        self.out6 = self.classifier(self.out6)

        return self.out6

def visFirstLayer():
    model = AlexNet().eval()
    W = model.conv1[0].weight.data.numpy()
    visWeights(W)
    return

if __name__ == '__main__':
    start = time.time()

    visFirstLayer()

    end = time.time()
    print('Time elapsed:', end-start)
