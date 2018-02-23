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
act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']

class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        self.conv1[0].weight = an_builtin.features[0].weight
        self.conv1[0].bias = an_builtin.features[0].bias

        self.conv2[0].weight = an_builtin.features[3].weight
        self.conv2[0].bias = an_builtin.features[3].bias

        self.conv3[0].weight = an_builtin.features[6].weight
        self.conv3[0].bias = an_builtin.features[6].bias

        self.conv4[0].weight = an_builtin.features[8].weight
        self.conv4[0].bias = an_builtin.features[8].bias

        self.conv5[0].weight = an_builtin.features[10].weight
        self.conv5[0].bias = an_builtin.features[10].bias

        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self.load_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        #x = self.classifier(x)
        return x

#______________________________ PART 10 _____________________________#
def part10():

    # convert images to 227x227
#    getData(act, (227, 227), download=False)

    # init model
    model = MyAlexNet()
    model.eval()

    # read an image
    im = imread('processed/227x227/alec_baldwin1.jpg')[:,:,:3]
    im = im - np.mean(im.flatten())
    im = im/np.max(np.abs(im.flatten()))
    im = np.rollaxis(im, -1).astype(np.float32)

    # turn the image into a numpy variable
    im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)

    # run the forward pass AlexNet prediction
    softmax = nn.Softmax()
    all_probs = softmax(model.forward(im_v)).data.numpy()[0]
    sorted_ans = np.argsort(all_probs)

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
