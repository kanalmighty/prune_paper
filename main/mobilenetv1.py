from torch import nn

#模型是这样的，输入3
class Mobile_Net(nn.Module):
    def __init__(self,num_class,cfg,origin_cfg):
        super(Mobile_Net, self).__init__()
        self.origin_cfg = origin_cfg
        self.cfg = cfg
        self.feature = nn.Sequential(
            Conv_Bn(3, self.cfg[0], 2),
            Conv_Dw(self.cfg[0], self.cfg[1], 1),
            Conv_Dw(self.cfg[1], self.cfg[2], 2),
            Conv_Dw(self.cfg[2], self.cfg[3], 1),
            Conv_Dw(self.cfg[3], self.cfg[4], 2),
            Conv_Dw(self.cfg[4], self.cfg[5], 1),
            Conv_Dw(self.cfg[5], self.cfg[6], 2),
            Conv_Dw(self.cfg[6], self.cfg[7], 1),
            Conv_Dw(self.cfg[7], self.cfg[8], 1),
            Conv_Dw(self.cfg[8], self.cfg[9], 1),
            Conv_Dw(self.cfg[9], self.cfg[10], 1),
            Conv_Dw(self.cfg[10], self.cfg[11], 1),
            Conv_Dw(self.cfg[11], self.cfg[12], 2),
            Conv_Dw(self.cfg[12], self.cfg[13], 1),
            nn.AvgPool2d(7),
        )
        self.classifier = nn.Linear(self.cfg[13], num_class)

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, self.cfg[13])
        x = self.classifier(x)
        return x


class Conv_Bn(nn.Module):
    def __init__(self, inp, oup,stride):
        super(Conv_Bn, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('conv1', nn.Conv2d(inp, oup, 3, stride, 1, bias=False))
        self.layers.add_module('norm1',  nn.BatchNorm2d(oup))
        self.layers.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)

class Conv_Dw(nn.Module):
        def __init__(self, inp, oup, stride):
            super(Conv_Dw, self).__init__()
            self.layers = nn.Sequential()
            self.layers.add_module('conv1_group', nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False))
            self.layers.add_module('norm1', nn.BatchNorm2d(inp))
            self.layers.add_module('relu', nn.ReLU(inplace=True))

            self.layers.add_module('conv2', nn.Conv2d(inp, oup, 1, 1, 0, bias=False))
            self.layers.add_module('norm2', nn.BatchNorm2d(oup))
            self.layers.add_module('relu', nn.ReLU(inplace=True))

        def forward(self, x):
            return self.layers(x)

def mobile_net_v1(num_class, cfg=None):
    origin_cfg = [32, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, 1024]
    cfg = origin_cfg if cfg is None else cfg
    return Mobile_Net(num_class, cfg=cfg, origin_cfg=origin_cfg)

if __name__ == '__main__':
    net = Mobile_Net()
    print(net.state_dict())