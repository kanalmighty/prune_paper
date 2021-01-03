import sys
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
    default='cifar100',
    choices=('cifar10', 'cifar100'),
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
    default='resnet_34_compare_0.9.pth',
    # default="none",
    help='load the model from the specified checkpoint')
parser.add_argument(
    '--ref_model',
    type=str,
    # default=None,
    default="66.82_resnet_100_0.9.pth",
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
    '--num_class',
    type=int,
    default='100')
parser.add_argument(
    '--arch',
    type=str,
    default='resnet_34',
    choices=('AlexNet', 'vgg_16_bn','resnet_34','vgg_19_bn','mobile_net_v1'),
    help='The architecture to prune')
args = parser.parse_args()


lr_decay_step = list(map(int, args.lr_decay_step.split(',')))



trainloader,testloader = get_loaders(args.dataset, args.data_dir,args.train_batch_size,args.eval_batch_size,args.arch)

project_root_path = os.path.abspath(os.path.dirname(__file__))
if sys.platform == 'linux':
    tmp_dir = '/content/drive/MyDrive/'
else:
    tmp_dir = os.path.join(project_root_path, 'tmp')


if not Path(tmp_dir).exists():
    os.mkdir(tmp_dir)
print(vars(args))

# Model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Training
def compare_train():
    if args.resume == 'none':
        save_path = os.path.join(tmp_dir, args.arch+'_compare.pth')
        model_state_pre_best = get_model(args.ref_model, device=device)
        best_acc = 0  # best test accuracy
        cfg = model_state_pre_best['cfg']
        net_pruned = eval(args.arch)(args.num_class, cfg=cfg)
        net_pruned.to(device)
        optimizer = optim.SGD(net_pruned.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:

        model_state_pre_best = get_model(args.resume, device=device)
        best_acc = model_state_pre_best['lasted_best_prec1']
        save_path = os.path.join(tmp_dir, args.resume)
        cfg = model_state_pre_best['cfg']
        net_pruned = eval(args.arch)(args.num_class, cfg=cfg)
        net_pruned.load_state_dict(model_state_pre_best['state_dict'])
        net_pruned.to(device)
        optimizer = optim.SGD(net_pruned.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer.load_state_dict(model_state_pre_best['optimizer'])

    end_epoch = 100
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch + 1, end_epoch):
        net_pruned.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            with torch.cuda.device(device):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = net_pruned(inputs)
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
        net_pruned.eval()
        num_iterations = len(testloader)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net_pruned(inputs)

                prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))

            print(
                'Epoch[{0}]({1}/{2}): '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iterations, top1=top1, top5=top5))

        if (top1.avg.item() > best_acc):
            print('当前测试精度%f大于当前层剪枝周期的最号精度%f,保存模型%s)' % (
            top1.avg.item(), best_acc, save_path))

            best_acc = top1.avg.item()

            model_state = {
                'state_dict': net_pruned.state_dict(),
                'lasted_best_prec1': round(best_acc, 2),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'cfg': cfg,
            }
            torch.save(model_state, save_path)












if __name__ == '__main__':
    compare_train()
    # drop_then_train()