# 本resnet结构选自：https://arxiv.org/pdf/1603.05027.pdf
# 是原作者在resnet上的更新版本(实际用的没有原始版本广，认可度有质疑)

from __future__ import absolute_import
import math
import torch.nn as nn
from .channel_selection import channel_selection


__all__ = ['resnet']

"""
preactivation resnet with bottleneck design.
"""

class Bottleneck(nn.Module):
    expansion = 4
    # self.inplanes = 16, 64, 128 (planes * 4)
    # planes = 16, 32, 64
    # stride=1, 2, 2
    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        # 从conv3看出，planes*4应该=inplances
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class resnet(nn.Module):
    def __init__(self, depth=164, dataset='cifar10', cfg=None):
        # depth=164 is too big to me. Let me change it to 20.
        super(resnet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9    # n = blocks = 2
        block = Bottleneck

        # cfg目的是针对prune后生成相应channel的新的network（cfg中的个数来源于bn.weight.data.gt(thre),定义于resprune.py）
        if cfg is None:
            # Construct config variable.
            cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]
            # 此行：拆掉最内层维度，使cfg变成一维list
            cfg = [item for sub_list in cfg for item in sub_list]

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg=cfg[0:3*n])
        self.layer2 = self._make_layer(block, 32, n, cfg=cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg=cfg[6*n:9*n], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.select = channel_selection(64 * block.expansion)   # select必须接在batchnorm后面
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)

        if dataset == 'cifar10':
            self.fc = nn.Linear(cfg[-1], 10)
        elif dataset == 'cifar100':
            self.fc = nn.Linear(cfg[-1], 100)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        """
        做resnet的基本组件(层，layer)
        :param block: 原始resnet有两种。Basic和Bottleneck。区别在于Basic适用于短的resnet(34层以下);bottleneck是50层以上的。
                      Basic是2个3x3 conv，Bottleneck是1x1+3x3+1x1。不过它们彼此channel是兼容的，所以其实可以混用。
                      这里为了简化，全部采用Bottleneck。
        :param planes: 根据3层(layer)，具体分别为: 16, 32, 64.
        :param blocks: 每层中多少个block，我用的是depth 20，所以blocks个数是2((depth - 2) // 9 = 2)
        :param cfg: 每次batchnorm需要保留的channel个数
        :param stride: layer1=1，layer2/3=2
        :return:
        """
        downsample = None
        # self.inplanes = 16, 64, 128 (planes * 4)
        # planes = 16, 32, 64
        # block.expansion = 4
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
