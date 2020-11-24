import torch
import torch.nn as nn
import torch.nn.functional as F




norm_mean, norm_var = 0.0, 1.0


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)





class ResNet(nn.Module):
    def __init__(self,num_classes,covcfg):
        super(ResNet, self).__init__()
        self.cfg = covcfg
        self.origin_cfg = [64, 64, 64,64,64,64,64,128,128,128,128,128,128,128,128,256,256,256,256,256,256,256,256,256,256,256,256,512,512,512,512]
        self.inplanes = 3
        # self.cfg.insert(0, self.inplanes)
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.feature99 = Resnet_Block_3(self.inplanes, self.cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.feature00 = Resnet_Block_3(self.cfg[0], self.cfg[1])
        self.feature01 = Resnet_Block_2(self.cfg[1], self.cfg[2])

        self.feature10 = Resnet_Block_3(self.cfg[2], self.cfg[3])
        self.feature11 = Resnet_Block_2(self.cfg[3], self.cfg[4])

        self.feature20 = Resnet_Block_3(self.cfg[4], self.cfg[5])
        self.feature21 = Resnet_Block_2(self.cfg[5], self.cfg[6])

        self.feature30 = Resnet_Block_3(self.cfg[6], self.cfg[7])
        self.feature31 = Resnet_Block_2(self.cfg[7], self.cfg[8])

        self.feature40 = Resnet_Block_3(self.cfg[8], self.cfg[9])
        self.feature41 = Resnet_Block_2(self.cfg[9], self.cfg[10])

        self.feature50 = Resnet_Block_3(self.cfg[10], self.cfg[11])
        self.feature51 = Resnet_Block_2(self.cfg[11], self.cfg[12])

        self.feature60 = Resnet_Block_3(self.cfg[12], self.cfg[13])
        self.feature61 = Resnet_Block_2(self.cfg[13], self.cfg[14])

        self.feature70 = Resnet_Block_3(self.cfg[14], self.cfg[15])
        self.feature71 = Resnet_Block_2(self.cfg[15], self.cfg[16])

        self.feature80 = Resnet_Block_3(self.cfg[16], self.cfg[17])
        self.feature81 = Resnet_Block_2(self.cfg[17], self.cfg[18])

        self.feature90 = Resnet_Block_3(self.cfg[18], self.cfg[19])
        self.feature91 = Resnet_Block_2(self.cfg[19], self.cfg[20])

        self.feature100 = Resnet_Block_3(self.cfg[20], self.cfg[21])
        self.feature101 = Resnet_Block_2(self.cfg[21], self.cfg[22])

        self.feature110 = Resnet_Block_3(self.cfg[22], self.cfg[23])
        self.feature111 = Resnet_Block_2(self.cfg[23], self.cfg[24])

        self.feature120 = Resnet_Block_3(self.cfg[24], self.cfg[25])
        self.feature121 = Resnet_Block_2(self.cfg[25], self.cfg[26])

        self.feature130 = Resnet_Block_3(self.cfg[26], self.cfg[27])
        self.feature131 = Resnet_Block_2(self.cfg[27], self.cfg[28])

        self.feature140 = Resnet_Block_3(self.cfg[28], self.cfg[29])

        self.feature150 = Resnet_Block_3(self.cfg[29], self.cfg[30])

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(self.cfg[30], num_classes)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, origin_x, conv_dropout_list=None):
        # self.original_cfg = [64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256,
        #                      256, 256, 256, 256, 256, 256, 256, 256, 512, 512, 512, 512]
        if conv_dropout_list is None:
            conv_dropout_list = [-1 for i in range(31)]
        from anal_utils import get_prune_mask, fix_shutcut_tensor
        origin_x = self.feature99(origin_x)
        origin_x = self.max_pool(origin_x)

        out_00 = self.feature00(origin_x)

        out_01 = self.feature01(out_00)

        # 开始有shutcut
        # resblock1层
        out_10 = self.feature10(out_01)
        out_11_tmp = self.feature11(out_10)
        out_01_fixed = fix_shutcut_tensor(out_01, self.origin_cfg[4])
        shutcut_01_prune_mask = get_prune_mask('fix', out_10[1], conv_dropout_list[4]).bool()
        out_11 = out_11_tmp + out_01_fixed[:, shutcut_01_prune_mask, :, :]
        # out_11 = out_11_tmp + out_01
        out_11 = self.relu(out_11)

        # resblock2层
        out_20 = self.feature20(out_11)
        out_21_tmp = self.feature21(out_20)
        out_31_fixed = fix_shutcut_tensor(out_11, self.origin_cfg[6])
        shutcut_11_prune_mask = get_prune_mask('fix', out_11[1], conv_dropout_list[6]).bool()
        out_21 = out_21_tmp + out_31_fixed[:, shutcut_11_prune_mask, :, :]
        # out_21 = out_21_tmp + out_11
        out_21 = self.relu(out_21)

        # resblock3层
        out_30 = self.feature30(out_21)
        out_31_tmp = self.feature31(out_30)
        out_21_fixed = fix_shutcut_tensor(out_21, self.origin_cfg[8])
        shutcut_21_prune_mask = get_prune_mask('fix', out_21_fixed[1], conv_dropout_list[8]).bool()
        out_31 = out_31_tmp + out_21_fixed[:, shutcut_21_prune_mask, :, :]
        # out_31 = out_31_tmp + out_21
        out_31 = self.relu(out_31)

        # resblock4层
        out_40 = self.feature40(out_31)
        out_41_tmp = self.feature41(out_40)
        # out_31_fixed = fix_shutcut_tensor(out_31, self.origin_cfg[10])
        # shutcut_30_prune_mask = get_prune_mask('fix', out_31[1], conv_dropout_list[10]).bool()
        # out_41 = out_41_tmp + out_31_fixed[:, shutcut_30_prune_mask, :, :]
        # out_41 = out_41_tmp + out_31
        out_41 = self.relu(out_41_tmp)

        # resblock5层
        out_50 = self.feature50(out_41)
        out_51_tmp = self.feature51(out_50)
        # out_41_fixed = fix_shutcut_tensor(out_41,self.origin_cfg[12])
        # shutcut_41_prune_mask = get_prune_mask('fix', out_41[1], conv_dropout_list[12]).bool()
        # out_51 = out_51_tmp + out_41_fixed[:, shutcut_41_prune_mask, :, :]
        # out_51 = out_51_tmp + out_41
        out_51 = self.relu(out_51_tmp)

        # resblock6层
        out_60 = self.feature60(out_51)
        out_61_tmp = self.feature61(out_60)
        # out_51_fixed = fix_shutcut_tensor(out_51, self.origin_cfg[14])
        # shutcut_51_prune_mask = get_prune_mask('fix', out_51[1], conv_dropout_list[14]).bool()
        # out_61 = out_61_tmp + out_51_fixed[:, shutcut_51_prune_mask, :, :]
        # out_61 = out_61_tmp + out_51
        out_61 = self.relu(out_61_tmp)

        # resblock7层
        out_70 = self.feature70(out_61)
        out_71_tmp = self.feature71(out_70)
        out_61_fixed = fix_shutcut_tensor(out_61, self.origin_cfg[16])
        shutcut_61_prune_mask = get_prune_mask('fix', out_61_fixed[1], conv_dropout_list[16]).bool()
        out_71 = out_71_tmp + out_61_fixed[:, shutcut_61_prune_mask, :, :]
        # out_71 = out_71_tmp + out_61
        out_71 = self.relu(out_71)

        # resblock8层
        out_80 = self.feature80(out_71)
        out_81_tmp = self.feature81(out_80)
        # out_71_fixed = fix_shutcut_tensor(out_71, self.origin_cfg[18])
        # shutcut_71_prune_mask = get_prune_mask('fix', out_71_fixed[1], conv_dropout_list[18]).bool()
        # out_81 = out_81_tmp + out_71_fixed[:, shutcut_71_prune_mask, :, :]
        # out_81 = out_81_tmp + out_71
        out_81 = self.relu(out_81_tmp)

        # resblock9层
        out_90 = self.feature90(out_81)
        out_91_tmp = self.feature91(out_90)
        # out_81_fixed = fix_shutcut_tensor(out_81, self.origin_cfg[20])
        # shutcut_81_prune_mask = get_prune_mask('fix', out_81_fixed[1], conv_dropout_list[20]).bool()
        # out_91 = out_91_tmp + out_81_fixed[:, shutcut_81_prune_mask, :, :]
        # out_91 = out_91_tmp + out_81
        out_91 = self.relu(out_91_tmp)

        # resblock10层
        out_100 = self.feature100(out_91)
        out_101_tmp = self.feature101(out_100)
        # out_91_fixed = fix_shutcut_tensor(out_91, self.origin_cfg[22])
        # shutcut_91_prune_mask = get_prune_mask('fix', out_91_fixed[1], conv_dropout_list[22]).bool()
        # out_101 = out_101_tmp + out_91_fixed[:, shutcut_91_prune_mask, :, :]
        # out_101 = out_101_tmp + out_91
        out_101 = self.relu(out_101_tmp)

        # resblock11层
        out_110 = self.feature110(out_101)
        out_111_tmp = self.feature111(out_110)
        # out_101_fixed = fix_shutcut_tensor(out_101, self.origin_cfg[24])
        # shutcut_101_prune_mask = get_prune_mask('fix', out_101_fixed[1], conv_dropout_list[24]).bool()
        # out_111 = out_111_tmp + out_101_fixed[:, shutcut_101_prune_mask, :, :]
        # out_111 = out_111_tmp + out_101
        out_111 = self.relu(out_111_tmp)

        # resblock12层
        out_120 = self.feature120(out_111)
        out_121_tmp = self.feature121(out_120)
        # out_111_fixed = fix_shutcut_tensor(out_111, self.origin_cfg[26])
        # shutcut_111_prune_mask = get_prune_mask('fix', out_111_fixed[1], conv_dropout_list[26]).bool()
        # out_121 = out_121_tmp + out_111_fixed[:, shutcut_111_prune_mask, :, :]
        # out_121 = out_121_tmp + out_111
        out_121 = self.relu(out_121_tmp)

        # resblock13层
        out_130 = self.feature130(out_121)
        out_131_tmp = self.feature131(out_130)
        out_121_fixed = fix_shutcut_tensor(out_121, self.origin_cfg[28])
        shutcut_121_prune_mask = get_prune_mask('fix', out_121_fixed[1], conv_dropout_list[28]).bool()
        out_131 = out_131_tmp + out_121_fixed[:, shutcut_121_prune_mask, :, :]
        # out_131 = out_121 + out_131_tmp
        out_131 = self.relu(out_131)

        # resblock14层
        out_140 = self.feature140(out_131)

        # resblock15层

        out_150 = self.feature150(out_140)
        out = nn.AvgPool2d(7)(out_150)

        y = self.avgpool(out)
        y = y.view(y.size(0), -1)
        out_fc = self.classifier(y)

        return out_fc


class Resnet_Block_3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Resnet_Block_3, self).__init__()
        self.layers = nn.Sequential()

        self.layers.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.layers.add_module('norm1', nn.BatchNorm2d(out_channels))
        self.layers.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class Resnet_Block_2(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Resnet_Block_2, self).__init__()
        self.layers = nn.Sequential()

        self.layers.add_module('conv2', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.layers.add_module('norm2', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return self.layers(x)


def resnet_34(num_class,cfg=None):
    resnet_default_cfg = [64,64,64,64,64,64,64,128,128,128,128,128,128,128,128,256,256,256,256,256,256,256,256,256,256,256,256,512,512,512,512]
# [64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256,
#                     256, 256, 256, 256, 256, 512, 512, 512, 511]
    resnet_cfg = cfg if cfg else resnet_default_cfg
    return ResNet(num_class,resnet_cfg)




if __name__ == '__main__':
    model = resnet_34()
    print(model)

