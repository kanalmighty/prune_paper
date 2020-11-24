import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from  main.anal_utils import *
import torchvision
import torchvision.transforms as transforms

import os
import argparse

from data import imagenet
from models import *

from mask import *
import utils


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument(
    '--model_path_1',
    type=str,
    default='D:\\datasets\\saved_model\\hrprine\\pruned_0.3_random.pth',
    help='dataset path')
parser.add_argument(
    '--model_path_2',
    type=str,
    default='D:\\datasets\\saved_model\\hrprine\\pruned_0.3_random.pth',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='D:\\datasets\\cifar-10-python\\cifar-10-batches-py',
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
    choices=('resnet_50','vgg_16_bn','resnet_56','resnet_110','densenet_40','googlenet'),
    help='The architecture to prune')

args = parser.parse_args()

#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_acc = 0  # best test accuracy



print_logger = utils.get_logger('D:\\PyCharmSpace\\HRank\\result\\tmp\\logger.log')


# Data
print_logger.info('==> Preparing data..')

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

testset = torchvision.datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False)
single_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
newcfg = [53, 40, 'M', 38, 40, 'M', 125, 88, 22, 'M', 245, 280, 52, 'M', 363, 442, 493, 512]#最后两个数字是l1的输入，输出和l2的输入
compress_rate = [0.95, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8]
def get_diffience_list(model_name_1, model_name_2):
    top1_1 = utils.AverageMeter()
    top5_1 = utils.AverageMeter()
    top1_2 = utils.AverageMeter()
    top5_2 = utils.AverageMeter()
    model_state_1 = get_model(model_name_1, device)
    model_state_2 = get_model(model_name_2, device)
    model_1 = VGG(num_classes=10,init_weights=False,cfg=model_state_1['cfg'])
    model_2 = VGG(num_classes=10, init_weights=False, cfg=model_state_2['cfg'])

    if 'state_dict' in model_state_1.keys():
        model_1.load_state_dict(model_state_1['state_dict'])
    else:
        model_2.load_state_dict(model_state_1)
    if 'state_dict' in model_state_2.keys():
        model_2.load_state_dict(model_state_2['state_dict'])
    else:
        model_2.load_state_dict(model_state_2)
    global best_acc
    model_1.cuda()
    model_1.eval()
    model_2.cuda()
    model_2.eval()
    sample_list  = []
    num_iterations = len(testloader)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_1 = model_1(inputs)
            outputs_2 = model_2(inputs)
            # print(targets)
            prec1_1, prec5_1 = utils.accuracy(outputs_1, targets, topk=(1, 5))
            prec1_2, prec5_2 = utils.accuracy(outputs_2, targets, topk=(1, 5))
            outputs_1 = torch.argmax(outputs_1, dim=1)
            outputs_2 = torch.argmax(outputs_2, dim=1)
            # targets = torch.argmax(targets, dim=1)

            compare_models = np.where(torch.eq(outputs_1, outputs_2).cpu().numpy() == 0)[0]
            compare_model1_target = np.where(torch.eq(outputs_1, targets).cpu().numpy() == 0)[0]
            compare_model2_target = np.where(torch.eq(outputs_2, targets).cpu().numpy() == 0)[0]
            if compare_model1_target.sum() == 0 and compare_model2_target.sum() > 0:
                print("1 is right")
            elif (compare_model2_target.sum() == 0 and compare_model1_target.sum() > 0):
                print("2 is right")
            compare = (torch.eq(outputs_1, outputs_2)  * torch.eq(outputs_1, targets) * torch.eq(outputs_2, targets))
            sample_list.append(torch.from_numpy(np.where(compare.cpu().numpy()==0)[0]))

            top1_1.update(prec1_1[0], inputs.size(0))
            top5_1.update(prec5_1[0], inputs.size(0))
            top1_2.update(prec1_2[0], inputs.size(0))
            top5_2.update(prec5_2[0], inputs.size(0))
        print_logger.info(
            '{top1.avg:.3f}'.format(top1=top1_1))
        print_logger.info(
            '{top1.avg:.3f}'.format(top1=top1_2))
    print(sample_list)
    #测试模型针对差异图片的输出
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            test_inputs, test_targets = inputs.to(device), targets.to(device)
            test_inputs = test_inputs.index_select(0, sample_list[batch_idx].cuda())
            test_targets = test_targets.index_select(0, sample_list[batch_idx].cuda())
            test_outputs_1 = model_1(test_inputs)
            test_outputs_2 = model_2(test_inputs)
            arg_max_outputs_1 = torch.argmax(test_outputs_1, dim=1)
            arg_max_outputs_2 = torch.argmax(test_outputs_2, dim=1)


            sample_list.append(list(np.where(compare.cpu().numpy()==0)))





if __name__ == '__main__':
   # get_diffience_list('vgg_full_baseline.pth', 'vgg_full_baseline.pth')
   get_diffience_list('fix313_9109.pth', 'fix372_9109.pth')


