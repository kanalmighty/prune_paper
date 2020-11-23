
import math
import torch
import torch.nn as nn
from collections import OrderedDict




# defaultcfg = [512, 512, 'M', 256, 256, 'M', 256, 256, 256, 'M', 128, 128, 128, 'M', 128, 64, 64, 64]
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]
convcfg = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]#卷积层的索引


class VGG(nn.Module):
    def __init__(self, num_classes, init_weights=True, cfg=None, origin_cfg=None):
        super(VGG, self).__init__()
        self.feature = nn.Sequential()
        self.origin_cfg = origin_cfg
        self.cfg = cfg if cfg is not None else self.origin_cfg

        # self.relucfg = relucfg
        # self.covcfg = convcfg
        self.feature = self.make_layers(self.cfg[:-1], True)
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(self.cfg[-2], self.cfg[-1])),
            ('norm1', nn.BatchNorm1d(self.cfg[-1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(self.cfg[-1], num_classes))
        ]))

        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=True):
        layers = nn.Sequential()
        in_channels = 3
        cnt = 0
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                cnt += 1

                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = v

        return layers

    def forward(self, x,conv_dropout_list=None):
        x = self.feature(x)

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg_16_bn(num_class, cfg=None):
    origin_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
    return VGG(num_class, cfg=cfg, origin_cfg=origin_cfg)


def vgg_19_bn(num_class, cfg=None):
    origin_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 512]
    return VGG(num_class, cfg=cfg, origin_cfg=origin_cfg)


class AlexNet(nn.Module):
    r""" An AlexNet model. Most easily loaded with the .from_name or .from_pretrained methods
    Args:
      global_params (namedtuple): A set of GlobalParams shared between blocks
    Examples:
        model = AlexNet.from_pretrained("alexnet")
    """

    def __init__(self, num_classes=100, cfg=None):
        super(AlexNet, self).__init__()
        conf_list = [64, 'M', 192, 384, 256, 'M', 256, 256]
        self.cfg = cfg if cfg is not None else conf_list
        self.feature = self.make_layers(self.cfg, True)
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2)
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.cfg[-1] * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )

        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(self.cfg[-1] * 6 * 6, 4096)),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(4096, 4096)),
            ('relu2', nn.ReLU(inplace=True)),
            ('linear3', nn.Linear(4096, num_classes))
        ]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def extract_feature(self, inputs):
        """ Returns output of the final convolution layer """
        x = self.feature(inputs)
        return x

    def make_layers(self, cfg, batch_norm=True):
        layers = nn.Sequential()
        in_channels = 3
        cnt = 0
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                cnt += 1

                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
                layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
                in_channels = v

        return layers

    def forward(self, inputs,conv_dropout_list):
        # See note [TorchScript super()]
        x = self.feature(inputs)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

