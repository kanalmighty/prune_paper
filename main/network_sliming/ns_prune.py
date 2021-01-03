import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import utils
from anal_utils import reset_kernel_by_list, get_model, get_conv_name_list, get_net_by_prune_dict, \
    get_conv_idx_by_name, search_by_conv_idx
from resnet34 import resnet_34
from network_sliming.vgg_ns import vgg_16_bn,vgg_19_bn
from utils import get_loaders
import numpy as np

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar100',
    choices=('cifar10','cifar100'),
    help='dataset')
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,
    help='Batch size for training.')
parser.add_argument(
    '--data_dir',
    type=str,
    default='D:\\datasets\\cifar-10-python\\cifar-10-batches-py\\',
    help='dataset path')
parser.add_argument('--eval_batch_size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--percent', type=float, default=0.2,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--resume', default='D:\\workspace\\prune_paper\\main\\tmp\\vgg16_ns_100_0.2.pth', type=str, metavar='PATH',
                    help='path to raw trained model (default: none)')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',
    choices=('AlexNet', 'vgg_16_bn','resnet_34','vgg_19_bn','mobile_net_v1'),
    help='The architecture to prune')
parser.add_argument('--save', default='D:\\workspace\\prune_paper\\main\\tmp', type=str, metavar='PATH')
parser.add_argument(
    '--num_class',
    type=int,
    default='100')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

trainloader,testloader = get_loaders(args.dataset, args.data_dir,args.train_batch_size,args.eval_batch_size,args.arch)

save_path = ''
if args.resume == "none":
    save_path = os.path.join(args.save, args.arch+'_ns.path')
else:
    save_path = os.path.join(args.save, args.resume)
model = eval(args.arch)(args.num_class)

model.cuda()
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = get_model(args.resume,device='cuda')
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.arch, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print(model)
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.clone()
        mask = weight_copy.abs().gt(thre.cuda()).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total

print('Pre-processing Successful!')


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test():
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    model.eval()
    correct = 0
    for data, target in testloader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))
    return correct / float(len(testloader.dataset))

test()

cfg.append(512)
# Make real prune
print(cfg)
newmodel = eval(args.arch)(args.num_class, cfg=cfg)
newmodel.cuda()

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        m1.weight.data = m0.weight.data[idx1].clone()
        m1.bias.data = m0.bias.data[idx1].clone()
        m1.running_mean = m0.running_mean[idx1].clone()
        m1.running_var = m0.running_var[idx1].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d} Out shape:{:d}'.format(idx0.shape[0], idx1.shape[0]))
        w = m0.weight.data[:, idx0, :, :].clone()
        w = w[idx1, :, :, :].clone()
        # if isinstance(m1, nn.Sequential) and isinstance(m1[0], nn.Linear):
        #     m1[0].weight.data = w.clone()
        # else:
        #     m1.weight.data = w.clone()
        m1.weight.data = w.clone()
        # m1.bias.data = m0.bias.data[idx1].clone()
    elif isinstance(m0, nn.Linear):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        m1.weight.data = m0.weight.data[:, idx0].clone()



print(newmodel)
model = newmodel
acc = test()

torch.save({'cfg': cfg,
            'state_dict': model.state_dict(),
            'best_prec1': acc*100,
            'epoch': checkpoint['epoch'],
            'optimizer': checkpoint['optimizer'],
            }, save_path)
