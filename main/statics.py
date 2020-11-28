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

from models import AlexNet

from models import *
from utils import get_loaders

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--model_name',
    type=str,
    default='66.43000030517578_feature.conv16.weight_vgg16_100_0.8.pth',
    help='dataset path')
parser.add_argument(
    '--data_dir',
    type=str,
    default='D:\\datasets\\cifar-10-python\\cifar-10-batches-py\\',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar100',
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
    default='100'),
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
    net = eval(args.arch)(args.num_class,cfg=cfg)

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
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = net(inputs)
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()
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
        print(top1.avg.item())
        end_time = time.time()
        print(end_time - start_time)
    # print("recall_macro = ",round(recall_macro/batch_count,3))
    # print("acc = ", round(acc / batch_count,3))
    # print("recall_micro = ", round(recall_micro / batch_count,3))
    # print("precision_macro = ", round(precision_macro / batch_count,3))
    # print("precision_micro = ", round(precision_micro / batch_count,3))

#cs,random中代表保留的百分比,
#o_mean,o_sum代表欧式距离近，数值小就剪枝
# def search(cfg_list, strategy,iteration):
#     print('cleanning model_states...')
#     ls = os.listdir('D:\\datasets\\saved_model\\hrprine\\model_states')
#     for i in ls:
#         c_path = os.path.join('D:\\datasets\\saved_model\\hrprine\\model_states', i)
#         os.remove(c_path)
#     for i in range(iteration):
#         # print_logger.info('strategy = %s, thd = %f' % (strategy, thd))
#         model_mean, model_var = get_prune_model('vgg_full_baseline.pth', cfg_list, strategy)
#         model_mean_list.append(model_mean)
#         model_var_list.append(model_var)
#         top1 = test('D:\\datasets\\saved_model\\hrprine\\tmp\\pruned_'+strategy+'.pth')
#         top1_list.append(top1)
        # os.rename('D:\\datasets\\saved_model\\hrprine\\tmp\\pruned_'+str(thd)+'_'+strategy+'.pth','D:\\datasets\\saved_model\\hrprine\\model_states\\pruned_'+str(thd)+'_'+strategy+'_'+str(top1)+'_'+str('nan')+'.pth')

if __name__ == '__main__':
    # cfg_list = [0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0, 1, 0]
    # model_mean_list = []
    # model_var_list = []
    # top1_list = []
    test()
    # test('vgg_new.pth')
    # search(cfg_list, 'random')
    # plt.figure(1)
    # plt.subplot(211)
    # plt.axis([80, 93, 0.0373, 0.03765 ])
    # plt.plot(top1_list,layer_zeros , 'bo')
    # plt.subplot(212)
    # plt.axis([80, 93,1.26735, 1.2677])
    # plt.plot(top1_list, model_mean_list, 'ro')
    # plt.show()


