

from __future__ import absolute_import
import os
import sys
import time
import logging
import datetime
import torch
from pathlib import Path

import torchvision
from torchvision import transforms


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('kd')
    # log_format = '%(asctime)s | %(message)s'
    log_format = '%(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def resore_cfg_maxpool(origin_cfg, pruned_cfg):
    for idx, i in enumerate(origin_cfg):
        if i == 'M':
            pruned_cfg.insert(idx, 'M')
    return pruned_cfg



def get_loaders(dataset,data_dir,train_batch_size, eval_batch_size):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == 'cifar10':

        trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                                transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False)
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True,
                                                 transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True)

        testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True,
                                                transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False)
    else:
        assert 1 == 0

    return trainloader, testloader
