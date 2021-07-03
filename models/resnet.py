from collections import namedtuple

import torch
import torch.nn as nn
import torchvision.models as models

"""Resnet50 Implemetention on https://github.com/pytorch/vision"""
"""Group convolution parts are deleted"""

def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation,
                     bias=False, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, norm_layer=None, downsample=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = norm_layer(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        identity = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)

        if self.downsample is not None:
            identity = self.downsample(x)

        y += identity
        y = self.relu(y)

        return y


class Resnet(nn.Module):
    def __init__(self, layers_num, num_classes=None, zero_init_residual=False, norm_layer=None, classify=False):
        super(Resnet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_channels = 64
        self.classify = classify

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers_num[0])
        self.layer2 = self._make_layer(128, layers_num[1], stride=2)
        self.layer3 = self._make_layer(256, layers_num[2], stride=2)
        self.layer4 = self._make_layer(512, layers_num[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        # self.sigmoid = nn.Sigmoid()

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.constant_(module.bn3.weight, 0)


    def _make_layer(self, out_channels, layer_num, block=Bottleneck, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, out_channels * block.expansion, stride),
                norm_layer(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, norm_layer, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, layer_num):
            layers.append(block(self.in_channels, out_channels, norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def load_pretrained_model(self, state_dict):
        required_state_dict = self.state_dict()
        for key in state_dict.keys():
            if key in required_state_dict.keys() and not key.startswith('fc'):
                required_state_dict[key] = state_dict[key]
        self.load_state_dict(required_state_dict)


    def forward(self, x):
        features = []

        x0 = self.conv1(x)           #256x256x64
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)        #128x128x64

        x1 = self.layer1(x0)         #128x128x256
        features.append(x1)

        x2 = self.layer2(x1)         #64x64x512
        features.append(x2)

        x3 = self.layer3(x2)         #32x32x1024
        features.append(x3)

        x4 = self.layer4(x3)         #16x16x2048
        features.append(x4)

        return features


class vgg16(nn.Module):
    def __init__(self, num_classes):
        super(vgg16, self).__init__()
        self.net = models.vgg16(num_classes=num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.net(x)
        y = self.sigmoid(y)
        return y


def resnet50(num_classes=None, classify=False, pretrained_model=None):
    net = Resnet([3, 4, 6, 3], num_classes=num_classes, classify=classify)
    if pretrained_model is not None:
        net.load_pretrained_model(torch.load(pretrained_model))
    return net
