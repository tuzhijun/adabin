'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
'''
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from utils.binarylib import AdaBin_Conv2d, Maxout

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class VGG_SMALL_1W1A(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_1W1A, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(128)
        self.nonlinear0 = Maxout(128)

        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = AdaBin_Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.nonlinear1 = Maxout(128)

        self.conv2 = AdaBin_Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.nonlinear2 = Maxout(256)

        self.conv3 = AdaBin_Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.nonlinear3 = Maxout(256)

        self.conv4 = AdaBin_Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.nonlinear4 = Maxout(512)

        self.conv5 = AdaBin_Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.nonlinear5 = Maxout(512)

        self.fc = nn.Linear(512*4*4, num_classes)

        self.apply(_weights_init)
        # self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, AdaBin_Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.nonlinear0(self.bn0(self.conv0(x)))

        x = self.conv1(x)
        x = self.pooling(x)
        x = self.nonlinear1(self.bn1(x))

        x = self.nonlinear2(self.bn2(self.conv2(x)))

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.nonlinear3(self.bn3(x))

        x = self.nonlinear4(self.bn4(self.conv4(x)))

        x = self.conv5(x)
        x = self.pooling(x)
        x = self.nonlinear5(self.bn5(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model