from pathlib import Path

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import tqdm
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from mobilenetv1 import mobile_net_v1
from torch import nn

import utils
from anal_utils import reset_kernel_by_list, get_model, get_conv_name_list, get_net_by_prune_dict, \
    get_conv_idx_by_name, search_by_conv_idx
from resnet34 import resnet_34
from models import *
from utils import get_loaders

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--data_dir',
    type=str,
    default='D:\\datasets\\cifar-10-python\\cifar-10-batches-py\\',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10','cifar100'),
    help='dataset')
parser.add_argument(
    '--lr',
    default=0.01,
    type=float,
    help='initial learning rate')
parser.add_argument(
    '--lr_decay_step',
    default='5,10',
    type=str,
    help='learning rate decay step')
parser.add_argument(
    '--resume',
    type=str,
    # default=None,
    default="mobile_net_v1_cifar10.pth",
    help='load the model from the specified checkpoint')
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,
    help='Batch size for training.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation.')
parser.add_argument(
    '--prune_rate',
    type=float,
    default=0.7,
    help='prune rate')
parser.add_argument(
    '--num_class',
    type=int,
    default='10')
parser.add_argument(
    '--arch',
    type=str,
    default='mobile_net_v1',
    choices=('AlexNet', 'vgg_16_bn','resnet_34','vgg_19_bn','mobile_net_v1'),
    help='The architecture to prune')
args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
lr_decay_step = list(map(int, args.lr_decay_step.split(',')))



trainloader,testloader = get_loaders(args.dataset, args.data_dir,args.train_batch_size,args.eval_batch_size)

project_root_path = os.path.abspath(os.path.dirname(__file__))
tmp_dir = os.path.join(project_root_path, 'tmp')


if not Path(tmp_dir).exists():
    os.mkdir(tmp_dir)
print(vars(args))

# Model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def prune_train():


    print('checkpoint %s exists,train from checkpoint' % args.resume)
    save_path = os.path.join(tmp_dir, args.resume)
    model_state = get_model(args.resume, device=device)


    conv_list = get_conv_name_list(model_state)
    conv_list_asc = conv_list.copy()
    conv_list.reverse()

    try:
        start_epoch = model_state['epoch']
    except KeyError:
        start_epoch = 0
    end_epoch = start_epoch + 25


    cudnn.benchmark = True
    print('\nEpoch: %d' % start_epoch)

    criterion = nn.CrossEntropyLoss()
    for conv_layername in conv_list:
        model_state_pre_best = get_model(args.resume, device=device)

        try:
            pruned_layers = model_state['pruned_layer']
        except KeyError:
            pruned_layers = dict()
        try:
            baseline_best_prec1 = model_state_pre_best['baseline_best_prec1']
        except KeyError:
            baseline_best_prec1 = model_state_pre_best['best_prec1']
            model_state_pre_best['baseline_best_prec1'] = baseline_best_prec1

        try:
            previous_lay_qualified_top1_mean = model_state_pre_best['previous_lay_qualified_top1_mean']
        except KeyError:
            model_state_pre_best['previous_lay_qualified_top1_mean'] = baseline_best_prec1

        if not 'lasted_best_prec1' in model_state_pre_best.keys():
            model_state_pre_best['lasted_best_prec1'] = model_state_pre_best['baseline_best_prec1']

        # 剪枝某一层时初始化当前层最好精度为0
        lasted_best_prec1 = 0
        cfg = model_state_pre_best['cfg']
        net_pre_best = eval(args.arch)(args.num_class, cfg=cfg)
        net_pre_best.load_state_dict(model_state_pre_best['state_dict'])
        optimizer = optim.SGD(net_pre_best.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer.load_state_dict(model_state_pre_best['optimizer'])
        print("获取上轮剪枝最佳模型,剪枝清单为%s,测试后准确率为%f" % (pruned_layers, lasted_best_prec1))
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()


        #如果已经剪枝则跳过下面的逻辑进行下一层的剪枝
        if conv_layername in pruned_layers.keys():
            continue
        #获取当前卷积层在所有卷积层总顺序派的索引号
        conv_prune_dict_cleaned = {name: -1 for name in conv_list_asc}
        conv_idx = get_conv_idx_by_name(conv_layername, conv_list_asc)
        print('%s在所有卷积层总顺序派的索引号为%d'%(conv_layername, conv_idx))
        print("处理%s,剪枝测试中" % conv_layername)
        # conv_prune_dict_cleaned[conv_layername] = {i if i%10==0 else 0 for i in range(512)}
        conv_prune_dict_cleaned[conv_layername], current_lay_qualified_top1_mean = search_by_conv_idx(model_state_pre_best,net_pre_best.origin_cfg, conv_layername, conv_idx, len(conv_list),testloader,args)

        print("剪枝字典为%s" % str(conv_prune_dict_cleaned))
        net_current_pruned, pruned_cfg = get_net_by_prune_dict(net_pre_best, args, conv_prune_dict_cleaned)
        net_current_pruned = net_current_pruned.to(device)
        pruned_layers[conv_layername] = conv_prune_dict_cleaned[conv_layername]
        total_drop_list = [list(v) if isinstance(v, list) else -1 for k, v in conv_prune_dict_cleaned.items()]

        total_drop_list_resnet34 = []

        try:
            for k, v in model_state['pruned_layer'].items():
                total_drop_list_resnet34.append(v)
        except KeyError:
            print('pruned_layer为空')

        for i in range(len(conv_list) - len(total_drop_list_resnet34)):
            total_drop_list_resnet34.append(-1)
        total_drop_list_resnet34.reverse()
        for epoch in range(start_epoch+1, end_epoch):
            net_current_pruned.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                with torch.cuda.device(device):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()
                    outputs = net_current_pruned(inputs,total_drop_list_resnet34)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # print(batch_idx,len(trainloader),
                    #              ' Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    #              % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            top1 = utils.AverageMeter()
            top5 = utils.AverageMeter()
            net_current_pruned.eval()
            num_iterations = len(testloader)
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net_current_pruned(inputs,total_drop_list_resnet34)

                    prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
                    top1.update(prec1[0], inputs.size(0))
                    top5.update(prec5[0], inputs.size(0))

                print(
                    'Epoch[{0}]({1}/{2}): '
                    'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                        epoch, batch_idx, num_iterations, top1=top1, top5=top5))

            if (top1.avg.item() > lasted_best_prec1):
                print('当前测试精度%f大于当前层剪枝周期的最号精度%f,baseline最好精度为%f,当前层剪枝精度平均%f,保存模型%s)'%(top1.avg.item(), lasted_best_prec1, baseline_best_prec1,current_lay_qualified_top1_mean,save_path))

                lasted_best_prec1 = top1.avg.item()

                model_state = {
                    'state_dict': net_current_pruned.state_dict(),
                    'baseline_best_prec1': baseline_best_prec1,
                    'lasted_best_prec1': round(lasted_best_prec1,2),
                    'epoch': epoch,
                    'previous_lay_qualified_top1_mean':current_lay_qualified_top1_mean,
                    'optimizer': optimizer.state_dict(),
                    'pruned_layer': pruned_layers,
                    'cfg': pruned_cfg,
                }
                torch.save(model_state, save_path)
                best_model_path = os.path.join(tmp_dir, str(lasted_best_prec1)+'_'+conv_layername+'.pth')
                torch.save(model_state, best_model_path)


if __name__ == '__main__':
    prune_train()