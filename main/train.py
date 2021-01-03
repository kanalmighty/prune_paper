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
from network_sliming.vgg_ns import *
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
    # type=str,
    # default="none",
    default='vgg16_ns_10_0.2.pth',
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
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('AlexNet', 'vgg_16_bn','resnet_34','vgg_19_bn','mobile_net_v1'),
    help='The architecture to prune')
parser.add_argument(
    '--num_class',
    type=int,
    default='10'),
parser.add_argument(
    '--drop_train',
    type=str,
    default=None,
    help='The architecture to prune')

args = parser.parse_args()

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
lr_decay_step = list(map(int, args.lr_decay_step.split(',')))


print(vars(args))

project_root_path = os.path.abspath(os.path.dirname(__file__))
trainloader,testloader = get_loaders(args.dataset, args.data_dir,args.train_batch_size,args.eval_batch_size,args.arch)

# Model
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


if sys.platform == 'linux':
    baseline_dir = '/content/drive/MyDrive/'
else:
    baseline_dir = os.path.join(project_root_path, 'baseline')

if not Path(baseline_dir).exists():
    os.mkdir(baseline_dir)



# Training
def train_baseline():

    if args.resume != "none":
        print('checkpoint %s exists,train from checkpoint' % args.resume)
        save_path = os.path.join(baseline_dir, args.resume)
        # model_state = torch.load(args.resume, map_location=device)
        model_state = get_model(args.resume, device=device)
        cfg = model_state['cfg']
        net = eval(args.arch)(args.num_class)
        current_model_best_acc = model_state['best_prec1']
        net.load_state_dict(model_state['state_dict'])
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer.load_state_dict(model_state['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        try:
            start_epoch = model_state['epoch']
        except KeyError:
            start_epoch = 0
        end_epoch = start_epoch + 280
    else:
        save_path = os.path.join(baseline_dir, args.arch+'_' + args.dataset+'.pth')
        current_model_best_acc = 0
        net = eval(args.arch)(args.num_class)
        cfg = net.cfg
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        start_epoch = 0
        end_epoch = 100

    net = net.to(device)
    cudnn.benchmark = True
    print('\nEpoch: %d' % start_epoch)
    fake_drop_out_list = [-1 for i in range(31)]

    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch+1, end_epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            with torch.cuda.device(device):
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                print(batch_idx,len(trainloader),
                             ' Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        net.eval()
        num_iterations = len(testloader)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)

                prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
                top1.update(prec1[0], inputs.size(0))
                top5.update(prec5[0], inputs.size(0))

            print(
                'Epoch[{0}]({1}/{2}): '
                'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                    epoch, batch_idx, num_iterations, top1=top1, top5=top5))

        if (top1.avg.item() > current_model_best_acc):
            current_model_best_acc = top1.avg.item()
            model_state = {
                'state_dict': net.state_dict(),
                'best_prec1': current_model_best_acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'cfg': cfg,
            }

            torch.save(model_state, save_path)

    print("=>Best accuracy {:.3f}".format(model_state['best_prec1']))












if __name__ == '__main__':
    train_baseline()
    # drop_then_train()