import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from datetime import datetime
import time
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torchstat import stat
import numpy as np
from  anal_utils import *
import torchvision
import torchvision.transforms as transforms
from sklearn import metrics
import os
import argparse

from network_sliming.vgg_ns import vgg_16_bn,vgg_19_bn
from utils import get_loaders

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--model_name',
    type=str,
    default='vgg16_ns_10_0.2_tuned.pth',
    help='dataset path')
parser.add_argument(
    '--data_dir',
    type=str,
    default='D:\\datasets\\cifar-10-python\\cifar-10-batches-py\\',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    help='dataset')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation.')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('resnet_34','vgg_16_bn','vgg_19_bn','alexnet','densenet_40','mobile_net_v1'),
    help='The architecture to prune')
parser.add_argument(
    '--num_class',
    type=int,
    default='10'),
args = parser.parse_args()
args.train_batch_size = 128
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_acc = 0  # best test accuracy



print_logger = utils.get_logger('D:\\PyCharmSpace\\HRank\\result\\tmp\\logger.log')


# Data
print_logger.info('==> Preparing data..')

trainloader,testloader = get_loaders(args.dataset, args.data_dir,args.train_batch_size,args.eval_batch_size,args.arch)
def test():
    model_state = get_model(args.model_name, device=device)
    cfg = model_state['cfg']
    net = eval(args.arch)(args.num_class, cfg=cfg)

    if 'state_dict' in model_state.keys():
        net.load_state_dict(model_state['state_dict'])
    else:
        net.load_state_dict(model_state)

    #处理
    conv_list = get_conv_name_list(model_state)
    conv_list.reverse()
    total_drop_list_resnet34 = []
    net.cuda()
    try:
        for k, v in model_state['pruned_layer'].items():
            total_drop_list_resnet34.append(v)
    except KeyError:
        print('pruned_layer为空')

    for i in range(len(conv_list) - len(total_drop_list_resnet34)):
        total_drop_list_resnet34.append(-1)
    total_drop_list_resnet34.reverse()
    total = sum([param.nelement() for param in net.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))  # 每一百万为一个单位
    batch_count = len(testloader)
    recall_macro = 0
    acc = 0
    recall_micro = 0
    precision_macro = 0
    precision_micro = 0
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    net.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs,total_drop_list_resnet34)

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            # y_true：存放的是测试集数据的真实标签，数组形状[test_size]，数据类型
            # int，如{0, 1, 2, 3, 4}
            # y_true = targets.cpu().numpy()

            # y_true_onehot：存放的是测试集数据的真实标签， 数组形状[test_size, num_class]（num_class = 5）; 数据类型
            # int（值为0或1），0，代表不属于，1代表属于；
            # y_true_onehot = np.eye(args.num_class)[targets.cpu().numpy()]

            # y_score: 存放的是分类模型计算出的概率，数组形状[test_size, num_class]（num_class = 5）; 数据类型float（值为计算出的概率）；
            # y_score = outputs

            # y_pred: 与y_true形式相同的数组，只不过内部的值是通过网络预测出来的类别。根据y_score得出的预测结果（类别）.

            # y_pred = torch.argmax(y_score,dim=1).cpu().numpy()
            # acc += accuracy_score(y_true, y_pred)



            # recall_macro += recall_score(y_true, y_pred, average='macro')

            # recall_micro += recall_score(y_true, y_pred, average='micro')


            # precision_macro += precision_score(y_true, y_pred, average='macro')

            # precision_micro += precision_score(y_true, y_pred, average='micro')

        end_time = time.time()
        print(end_time - start_time)
    print(top1.avg.item())
    # print("recall_macro = ",round(recall_macro/batch_count,3))
    # print("acc = ", round(acc / batch_count,3))
    # print("recall_micro = ", round(recall_micro / batch_count,3))
    # print("precision_macro = ", round(precision_macro / batch_count,3))
    # print("precision_micro = ", round(precision_micro / batch_count,3))


if __name__ == '__main__':

    test()



