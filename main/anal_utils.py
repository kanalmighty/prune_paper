import copy
import os
import random
import matplotlib.pyplot as plt
import datetime
import glob
import torch
import re
import numpy as np
from numpy import mean
from torchstat import stat
import utils as utils
import torch.nn.functional as F
from mobilenetv1 import mobile_net_v1

from models import *
from resnet34 import resnet_34
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
layer_mean = []
layer_zeros=[]
total_var = []
def get_model(model_name, device):
    project_root_path = os.path.abspath(os.path.dirname(__file__))
    pth_path = project_root_path + os.sep + '*' + os.sep + '*.pth'
    for file in glob.glob(pth_path):
        if model_name in file:
            model = torch.load(file, map_location=device)
            return model

    raise Exception('no such %s model in %s'%(model_name,pth_path))



def get_parameter_rank(model):
    for p in model.parameter():
        print(torch.matrix_rank())


def read_rank_file(idx):
    rank_ndarray = np.load('D:\\PyCharmSpace\\HRank\\rank_conv\\vgg_16_bn\\rank_conv' + str(idx)+'.npy')
    return rank_ndarray

def get_difference_matrix(input, s_type):
    kernel_number, kernel_channel, w, h = input.shape[0],input.shape[1],input.shape[2],input.shape[3]
    output_tensor = torch.zeros(kernel_number, kernel_number)
    if s_type == 'cs':
        input = input.view(kernel_number, -1)

        for idx,kernel_tensor in enumerate(input):
            output_tensor[idx] = torch.cosine_similarity(kernel_tensor, input, dim=-1)
    #cs_tensor的dim1代表第几个kernel,dim2代表其与第几个kernel的相似度
    elif s_type == 'o':
        for idx, kernel_tensor in enumerate(input):
            kernel_tensor_e = kernel_tensor.expand(kernel_number,kernel_channel,w,h)
            kernel_tensor_v = kernel_tensor_e.view(kernel_number, -1)
            input_v = input.view(kernel_number, -1)
            output_tensor[idx] = F.pairwise_distance(kernel_tensor_v, input_v, p=2)
    return output_tensor

def get_model_statics(model_name):
    compress_rate = [0.95, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8]
    model_state = torch.load('D:\\datasets\\saved_model\\hrprine\\' + model_name, map_location=device)
    net = VGG(num_classes=10, init_weights=False, cfg=None, compress_rate=compress_rate)
    if 'state_dict' in model_state.keys():
        net.load_state_dict(model_state['state_dict'])
    else:
        net.load_state_dict(model_state)

    print(stat(net,(3,32,32)))



# def get_cs_mask_model(prtrained_model_name, prune_thres):
#     # compress_rate = [0.95, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8]
#     all_parameter = get_model(prtrained_model_name, device=device)['state_dict']
#     # model = VGG(num_classes=10, init_weights=False, cfg=None, compress_rate=compress_rate)
#     # print(model)
#
#     # a = conv0_parameter['feature.conv0.weight'].shape[0]
#     #
#     # b = conv0_parameter['feature.conv0.weight'].shape[1]
#     # c = torch.tensor([torch.matrix_rank(conv0_parameter['feature.conv0.weight'][i,j,:,:]).item() for i in range(a) for j in range(b)])
#     # 分析训练好的模型的kernel是否线性相关
#     p_dict = {}
#     conv_pattern = re.compile(r'feature.conv\d+.weight')
#     bias_pattern = re.compile(r'feature.conv\d+.bias')
#     for name, parameter in all_parameter.items():  # 循环一次代表一个conv层
#         mask_parameter = torch.ones_like(parameter)
#         if conv_pattern.match(name):
#
#
#             cs_matrix = get_difference_matrix(parameter, 'cs')#矩阵第一行代表一个向量与其他向量的相似度
#             cs_matrix_sum = cs_matrix.sum(dim=1)#汇总，记录每一个向量与矩阵中其他向量的相似度汇总值
#             global t1
#             t1 = cs_matrix_sum > prune_thres#相似度大于阈值说明冗余可以剪枝
#             t2 = t1.unsqueeze(1).unsqueeze(2).unsqueeze(3)
#             t3 = t2.repeat(1, mask_parameter.shape[1], mask_parameter.shape[2], mask_parameter.shape[3])
#             mask_parameter[t3] = 0
#             # mask_parameter = np.random.shuffle(mask_parameter.cpu().numpy()).from_numpy().cuda()
#             print('%s 减掉了 %f'%(name, t1.int().sum().item()/mask_parameter.shape[0]))
#             all_parameter[name] = parameter * mask_parameter
#         elif bias_pattern.match(name):
#             all_parameter[name] = parameter*(~t1).float().cuda()
#     torch.save(all_parameter, 'D:\\datasets\\saved_model\\hrprine\\cs_mask_prune_model.pth')
#para.requires_grad = False
def reset_kernel_by_list(model,parameter_dict):
    for layer, freeze_list in parameter_dict.items():
        for kernel_idx in freeze_list:
            # kernel_shape = model.state_dict()[layer][kernel_idx].size()
            model.state_dict()[layer][kernel_idx] = torch.zeros_like(model.state_dict()[layer][kernel_idx])
        # for idx, kernel in enumerate(model.state_dict()[layer]):
        #     if idx in freeze_list:
        #         kernel = torch.zeros_like(kernel)
    return model




def get_prune_mask(strategy, parameter, prune_thres):

    if strategy == 'cs_sum':
        cs_matrix = get_difference_matrix(parameter, 'cs')  # 矩阵第一行代表一个向量与其他向量的相似度
        cs_matrix_sum = cs_matrix.sum(dim=1)  # 汇总，记录每一个向量与矩阵中其他向量的相似度汇总值
        current_output_channel_mask = cs_matrix_sum > prune_thres  # 相似度大于阈值说明冗余可以剪枝
        current_output_channel_mask = ~current_output_channel_mask
    elif strategy == 'cs_mean':
        cs_matrix = get_difference_matrix(parameter, 'cs')   # 矩阵第一行代表一个向量与其他向量的相似度
        cs_matrix_sum = cs_matrix.sum(dim=1)  # 汇总，记录每一个向量与矩阵中其他向量的相似度汇总值
        current_output_channel_mask = cs_matrix_sum > prune_thres  # 相似度大于阈值说明冗余可以剪枝
        current_output_channel_mask = ~current_output_channel_mask
    elif strategy == 'thd':

        current_output_channel_mask = [1 if random.uniform(0, 1) > prune_thres else 0 for i in range(parameter.shape[0])]
        current_output_channel_mask = torch.tensor(current_output_channel_mask)
    elif strategy == 'random':

        prune_num = prune_thres
        # ones_number = np.random.randint(1, keep_num)
        np_ones = np.ones(parameter.shape[0] - prune_num)
        zeros_number = np.zeros(prune_num)
        mask = np.r_[np_ones, zeros_number]

        np.random.shuffle(mask)
        if prune_num > 0:
            layer_zeros.append(np.argwhere(mask == 0).squeeze().item())
        current_output_channel_mask = torch.tensor(mask)

    elif strategy == 'o_sum':
        dis_matrix = get_difference_matrix(parameter, 'o')
        dis_matrix_sum = dis_matrix.sum(dim=1)  # 汇总，记录每一个向量与矩阵中其他向量的相似度汇总值
        current_output_channel_mask = dis_matrix_sum < prune_thres  # 相似度大于阈值说明冗余可以剪枝
        current_output_channel_mask = ~current_output_channel_mask
    elif strategy == 'o_mean':
        dis_matrix = get_difference_matrix(parameter, 'o')
        dis_matrix_mean = dis_matrix.mean(dim=1)  # 汇总，记录每一个向量与矩阵中其他向量的相似度汇总值
        # print_logger.info(round(dis_matrix_mean.mean().item(), 3))
        current_output_channel_mask = dis_matrix_mean < prune_thres  # 相似度大于阈值说明冗余可以剪枝
        current_output_channel_mask = ~current_output_channel_mask
    elif strategy == 'o_mean_test':
        # dis_matrix = get_difference_matrix(parameter, 'o')
        kernel_num = parameter.shape[0]
        keep_v = np.ones(kernel_num-prune_thres)
        drop_v = np.zeros(prune_thres)
        mask = np.r_[keep_v, drop_v]

        np.random.shuffle(mask)
        #记录裁剪位置
        # print_logger.info((np.argwhere(mask==0)).squeeze())
        current_output_channel_mask = torch.tensor(mask)
        # 获取剪枝后的相似度均值,只记录不处理
        parameter = parameter[current_output_channel_mask.bool()]
        dis_matrix = get_difference_matrix(parameter, 'o')
        dis_matrix_mean = dis_matrix.mean(dim=1)  # 汇总，记录每一个向量与矩阵中其他向量的相似度汇总值

        #记录这个层的均值和方差
        layer_mean.append(dis_matrix_mean.mean())
        # 记录裁剪以后的mean
        # print_logger.info(round(dis_matrix_mean.mean().item(),3))
    elif strategy == 'fix':
        # dis_matrix = get_difference_matrix(parameter, 'o')
        kernel_num = parameter.shape[0]
        mask = np.ones(kernel_num)
        if isinstance(prune_thres, list):
            prune_thres = np.array(prune_thres,dtype=int)
            mask[prune_thres] = 0


        #记录裁剪位置
        # print_logger.info((np.argwhere(mask==0)).squeeze())
        current_output_channel_mask = torch.tensor(mask)
        # 获取剪枝后的相似度均值,只记录不处理

    return current_output_channel_mask
def get_conv_idx_by_name(conv_layername, conv_list):
    conv_idx = 0
    for name in conv_list:
        if name != conv_layername:
            conv_idx += 1
        else:

            break
    return conv_idx

# 根据指定conv名称，返回冗余卷积核序号
def search_by_conv_idx(model_state_dict, origin_cfg,conv_name, conv_idx, conv_list_length,testloader,args):
    # drop_out_dict = {}
    # 卷积层drop_out_list
    conv_dropout_list_resnet34 =[]
    try:
        for k,v in model_state_dict['pruned_layer'].items():
            conv_dropout_list_resnet34.append(v)
    except KeyError:
        print('pruned_layer为空')


    for i in range(conv_list_length - len(conv_dropout_list_resnet34)):
        conv_dropout_list_resnet34.append(-1)
    conv_dropout_list_resnet34.reverse()
    conv_dropout_list = [-1 for i in range(conv_list_length)]#resnet34 需要+1
    current_lay_qualified_top1_list = []
    origin_kernel_num = model_state_dict['state_dict'][conv_name].shape[0]
    pruned_model_state_dict = {
        # 'scheduler': model_state['scheduler'],
        'optimizer': model_state_dict['optimizer'],
        'epoch': model_state_dict['epoch'],
        'lasted_best_prec1': model_state_dict['lasted_best_prec1'],
        'previous_lay_qualified_top1_mean': model_state_dict['previous_lay_qualified_top1_mean'],
        'baseline_best_prec1': model_state_dict['baseline_best_prec1']
    }
    kernel_dropout_set = set()
    for kernel_idx in range(origin_kernel_num):
        #获取卷积核数据
        kernel_mean,kernel_sum,kernel_var,kernel_norm = get_kernel_analyse_by_id(args.resume,conv_name,kernel_idx)
        # print('共%d个卷积核，当前尝试删除第%d个卷积核,sum为%f,mean为%f,norm为%f,var为%f' % (origin_kernel_num, kernel_idx, kernel_mean, kernel_sum, kernel_norm,kernel_var))
        print('%f,%f,%f,%f' % (kernel_mean, kernel_sum, kernel_norm, kernel_var))
        conv_dropout_list[conv_idx] = [kernel_idx]
        conv_dropout_list_resnet34[conv_idx] = [kernel_idx]
        # print('输入get_pruned_model的dropout配置为' + str(conv_dropout_list))
        pruned_model_state = get_prune_model(model_state_dict['state_dict'], conv_dropout_list, 'fix',args)
        pruned_model_state_dict['state_dict'] = pruned_model_state['state_dict']
        pruned_model_state_dict['cfg'] = utils.resore_cfg_maxpool(origin_cfg, pruned_model_state['cfg'])
        # top1 = random.randint(-20,30)
        top1 = test_model(pruned_model_state_dict, testloader, args, conv_dropout_list_resnet34)
        # for i in pm.dropout_index:
        #     if isinstance(i, list):
        #         drop_outs.append(i[0].item())
        prune_threshold = model_state_dict['lasted_best_prec1']
        print(top1)
        if top1 > prune_threshold:
            current_lay_qualified_top1_list.append(top1)
            kernel_dropout_set.add(kernel_idx)
    # drop_out_dict[conv_name] = kernel_dropout_set
    kernel_dropout_list = list(kernel_dropout_set)
    kernel_dropout_list_rated =[]
    # print('获取%s层的冗余卷积核集合%s' % (conv_name, kernel_dropout_set))
    if args.prune_rate:
        #应剪枝个数
        to_prune_kernel_num = origin_kernel_num - int(origin_kernel_num * args.prune_rate)
        #冗余卷积列表长度len(kernel_dropout_list)
        if len(kernel_dropout_list) > to_prune_kernel_num:
            print("冗余卷积核列表长度为%d,根据剪枝率删减剪枝列表长度为%d"%(len(kernel_dropout_list), to_prune_kernel_num))
            kernel_dropout_list_rated = kernel_dropout_list[:to_prune_kernel_num]
        else:
            kernel_dropout_list_rated = kernel_dropout_list

    return -1 if len(kernel_dropout_list_rated)==0 else kernel_dropout_list_rated, mean(current_lay_qualified_top1_list)


#获取所有卷积层名称
def get_conv_name_list(model_state_dict):
    conv_list= []
    conv_group_pattern = re.compile(r'feature.*conv.*group.weight')

    conv_pattern = re.compile(r'feature.*conv.*weight')
    for name, parameter in model_state_dict['state_dict'].items():
        if conv_pattern.match(name) and conv_group_pattern.match(name) is None:
            conv_list.append(name)
    return conv_list

def get_parameter_by_list(prtrained_model_name, thres_cfg):
    newcfg = []
    model_state = get_model(prtrained_model_name, device=device)

    all_parameter = model_state['state_dict']
    conv_pattern = re.compile(r'feature.*conv.*weight')
    idx = 0
    res_dict = {}
    for name, parameter in all_parameter.items():  # 循环一次代表一个conv层
        if conv_pattern.match(name):
            if isinstance(thres_cfg[idx], list):
                res_dict[name] = parameter.index_select(0, torch.Tensor(thres_cfg[idx]).long().cuda())
            idx += 1

    state = {
        'state_dict': res_dict,
        'scheduler': model_state['scheduler'],
        'optimizer': model_state['optimizer'],
        'epoch': model_state['epoch'],
        'cfg': newcfg
    }

    return state


def test_model(model_state, testloader, args, conv_dropout_list):
    net = eval(args.arch)(args.num_class, cfg=model_state['cfg'])
    net = net.cuda()
    net.load_state_dict(model_state['state_dict'])
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs, conv_dropout_list)

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
    prune_threshold = model_state['lasted_best_prec1']
    if top1.avg.item() > prune_threshold:
       print("top1为%f,阈值为%f,drop该卷积核"%(top1.avg.item(),prune_threshold))
    return round(top1.avg.item(), 2)

def get_prune_model(pretrained_model, thres_cfg, strategy,args):
    newcfg = []

    if isinstance(pretrained_model, str):
        model_state = get_model(pretrained_model, device=device)
    else:
        model_state = pretrained_model

    all_parameter = model_state.copy()

    # 分析训练好的模型的kernel是否线性相关
    p_dict = {}


    conv_pattern = re.compile(r'feature.*conv.*weight')
    conv_group_pattern = re.compile(r'feature.*conv.*group.weight')
    bias_pattern = re.compile(r'feature.*conv.*bias')
    norm_pattern = re.compile(r'feature.*norm.*')
    linear1_pattern = re.compile(r'classifier.*.*')  # 处理分类器
    #获取第一个卷积层的输入通道
    previous_output_mask = torch.ones(3).bool()
    current_output_channel_mask = []
    idx = 0
    for name, parameter in all_parameter.items():  # 循环一次代表一个conv层
        if conv_pattern.match(name):
            try:
                prune_thres = thres_cfg[idx]
            except IndexError:
                prune_thres = thres_cfg[0]




            #如果不是分组卷积
            if conv_group_pattern.match(name) is None:
                current_output_channel_mask = get_prune_mask(strategy, parameter, prune_thres)
                current_output_channel_mask = current_output_channel_mask.bool()
                parameter = parameter[current_output_channel_mask]
                parameter = parameter[:, previous_output_mask]
                idx += 1
                newcfg.append(current_output_channel_mask.sum().item())
            all_parameter[name] = parameter
            previous_output_mask = current_output_channel_mask
        elif (bias_pattern.match(name) or norm_pattern.match(name)) and 'num_batches_tracked' not in name:
            all_parameter[name] = parameter[current_output_channel_mask]
            previous_output_mask = current_output_channel_mask
        elif linear1_pattern.match(name) and 'num_batches_tracked' not in name and 'bias' not in name:
            if 'linear1.weight' in name and args.arch is not 'AlexNet':
                all_parameter[name] = parameter[:, current_output_channel_mask]
                previous_output_mask = current_output_channel_mask
                newcfg.append(512)
            elif 'linear1.weight' in name and args.arch =='AlexNet':
                # drop_index = np.where(current_output_channel_mask==0)[0]
                # expanded_current_output_channel_mask = torch.ones(len(current_output_channel_mask)*36,dtype=int)
                # expanded_current_output_channel_mask[drop_index*36: drop_index*36+36] = 0
                drop_index = np.where(current_output_channel_mask == 0)[0]
                expand_tensor = torch.ones(36, 1)
                expanded_current_output_channel_mask = (current_output_channel_mask.long() * expand_tensor.long()).t().reshape(1, -1).squeeze()
                all_parameter[name] = parameter[:, expanded_current_output_channel_mask.bool()]
                previous_output_mask = current_output_channel_mask

            elif (args.arch == 'resnet_56' or args.arch == 'resnet_34') and 'bias' not in name:
                all_parameter[name] = parameter[:, current_output_channel_mask]
                previous_output_mask = current_output_channel_mask
            else:
                all_parameter[name] = parameter[:, current_output_channel_mask]
                previous_output_mask = ~current_output_channel_mask

    # vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512] 最后两个数字是l1的输入，输出和l2的输入
    # newcfg.insert(2, 'M')
    # newcfg.insert(5, 'M')
    # newcfg.insert(9, 'M')
    # newcfg.insert(13, 'M')

    # vgg19_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 512]
    # newcfg.insert(2, 'M')
    # newcfg.insert(5, 'M')
    # newcfg.insert(10, 'M')
    # newcfg.insert(15, 'M')

    # alexnet_cfg = [64, 'M', 192, 384, 256, 'M', 256, 256]
    # newcfg.insert(1, 'M')
    # newcfg.insert(5, 'M')

    #resnet没有最大池化层

    # resnet没有最大池化层
    # newcfg.insert(0, 64)
    #resnet34配置

    state = {
        'state_dict': all_parameter,
        # # 'scheduler': model_state['scheduler'],
        # 'optimizer': model_state['optimizer'],
        # 'epoch': model_state['epoch'],
        'cfg': newcfg
        # 'best_prec1': model_state['best_prec1']
    }
    layer_mean_np = np.array(layer_mean)
    # model_mean = round(layer_mean_np.mean().item(),5)
    # model_var = round(layer_mean_np.var().item(),5)
    # print_logger.info('saving model dict to D:\\datasets\\saved_model\\hrprine\\tmp\\pruned_' + str(prune_thres) + '_' + strategy +'.pth')
    return state


def get_first_output_mask(args):
    if args.arch == 'resnet_34':
        previous_output_mask = torch.ones(3).bool()
    else:
        previous_output_mask = torch.ones(3).bool()
    return previous_output_mask


def get_net_by_prune_dict(net, args, drop_dict):
    model_state = net.state_dict()
    print('dropping false kernels and re-train')
    drop_list = [list(v) if isinstance(v, list) else -1 for k, v in drop_dict.items()]
    # 把字典处理成数组
    model_state_pruned = get_prune_model(model_state, drop_list, 'fix', args)
    restored_cfg = utils.resore_cfg_maxpool(net.origin_cfg, model_state_pruned['cfg'])
    pruned_net = eval(args.arch)(args.num_class, cfg=restored_cfg)
    pruned_net.load_state_dict(model_state_pruned['state_dict'])
    # net = reset_kernel_by_list(net, drop_list)
    net = pruned_net.cuda()
    return net, model_state_pruned['cfg']

def clean_status(model_dict,dict_name):
    if dict_name in model_dict:
        del model_dict[dict_name]
    torch.save(model_dict, 'D:\\datasets\\saved_model\\hrprine\\tmp\\vgg_pruning_91.390.pth')


def change_model_dict():
    model = get_model('D:\\datasets\\saved_model\\hrprine\\tmp\\resnet34_pruning.pth',device=device)
    model['state_dict']['feature131.layers.conv2.bias'] = model['state_dict']['feature131.layers.tconv2.bias']
    del model['state_dict']['feature131.layers.tconv2.bias']
    model['state_dict']['feature121.layers.conv2.bias'] = model['state_dict']['feature121.layers.tconv2.bias']
    del model['state_dict']['feature121.layers.tconv2.bias']
    model['state_dict']['feature111.layers.conv2.bias'] = model['state_dict']['feature111.layers.tconv2.bias']
    del model['state_dict']['feature111.layers.tconv2.bias']
    model['state_dict']['feature101.layers.conv2.bias'] = model['state_dict']['feature101.layers.tconv2.bias']
    del model['state_dict']['feature101.layers.tconv2.bias']
    model['state_dict']['feature91.layers.conv2.bias'] = model['state_dict']['feature91.layers.tconv2.bias']
    del model['state_dict']['feature91.layers.tconv2.bias']
    model['state_dict']['feature81.layers.conv2.bias'] = model['state_dict']['feature81.layers.tconv2.bias']
    del model['state_dict']['feature81.layers.tconv2.bias']
    model['state_dict']['feature71.layers.conv2.bias'] = model['state_dict']['feature71.layers.tconv2.bias']
    del model['state_dict']['feature71.layers.tconv2.bias']
    model['state_dict']['feature61.layers.conv2.bias'] = model['state_dict']['feature61.layers.tconv2.bias']
    del model['state_dict']['feature61.layers.tconv2.bias']
    model['state_dict']['feature51.layers.conv2.bias'] = model['state_dict']['feature51.layers.tconv2.bias']
    del model['state_dict']['feature51.layers.tconv2.bias']
    model['state_dict']['feature41.layers.conv2.bias'] = model['state_dict']['feature41.layers.tconv2.bias']
    del model['state_dict']['feature41.layers.tconv2.bias']
    model['state_dict']['feature31.layers.conv2.bias'] = model['state_dict']['feature31.layers.tconv2.bias']
    del model['state_dict']['feature31.layers.tconv2.bias']
    model['state_dict']['feature21.layers.conv2.bias'] = model['state_dict']['feature21.layers.tconv2.bias']
    del model['state_dict']['feature21.layers.tconv2.bias']
    model['state_dict']['feature11.layers.conv2.bias'] = model['state_dict']['feature11.layers.tconv2.bias']
    del model['state_dict']['feature11.layers.tconv2.bias']
    model['state_dict']['feature01.layers.conv2.bias'] = model['state_dict']['feature01.layers.tconv2.bias']
    del model['state_dict']['feature01.layers.tconv2.bias']

    model['state_dict']['feature131.layers.conv2.weight'] = model['state_dict']['feature131.layers.tconv2.weight']
    del model['state_dict']['feature131.layers.tconv2.weight']
    model['state_dict']['feature121.layers.conv2.weight'] = model['state_dict']['feature121.layers.tconv2.weight']
    del model['state_dict']['feature121.layers.tconv2.weight']
    model['state_dict']['feature111.layers.conv2.weight'] = model['state_dict']['feature111.layers.tconv2.weight']
    del model['state_dict']['feature111.layers.tconv2.weight']
    model['state_dict']['feature101.layers.conv2.weight'] = model['state_dict']['feature101.layers.tconv2.weight']
    del model['state_dict']['feature101.layers.tconv2.weight']
    model['state_dict']['feature91.layers.conv2.weight'] = model['state_dict']['feature91.layers.tconv2.weight']
    del model['state_dict']['feature91.layers.tconv2.weight']
    model['state_dict']['feature81.layers.conv2.weight'] = model['state_dict']['feature81.layers.tconv2.weight']
    del model['state_dict']['feature81.layers.tconv2.weight']
    model['state_dict']['feature71.layers.conv2.weight'] = model['state_dict']['feature71.layers.tconv2.weight']
    del model['state_dict']['feature71.layers.tconv2.weight']
    model['state_dict']['feature61.layers.conv2.weight'] = model['state_dict']['feature61.layers.tconv2.weight']
    del model['state_dict']['feature61.layers.tconv2.weight']
    model['state_dict']['feature51.layers.conv2.weight'] = model['state_dict']['feature51.layers.tconv2.weight']
    del model['state_dict']['feature51.layers.tconv2.weight']
    model['state_dict']['feature41.layers.conv2.weight'] = model['state_dict']['feature41.layers.tconv2.weight']
    del model['state_dict']['feature41.layers.tconv2.weight']
    model['state_dict']['feature31.layers.conv2.weight'] = model['state_dict']['feature31.layers.tconv2.weight']
    del model['state_dict']['feature31.layers.tconv2.weight']
    model['state_dict']['feature21.layers.conv2.weight'] = model['state_dict']['feature21.layers.tconv2.weight']
    del model['state_dict']['feature21.layers.tconv2.weight']
    model['state_dict']['feature11.layers.conv2.weight'] = model['state_dict']['feature11.layers.tconv2.weight']
    del model['state_dict']['feature11.layers.tconv2.weight']
    model['state_dict']['feature01.layers.conv2.weight'] = model['state_dict']['feature01.layers.tconv2.weight']
    del model['state_dict']['feature01.layers.tconv2.weight']
    model['state_dict']  = sorted(model['state_dict'].items(), key=lambda x:x[0])
    torch.save(model, 'D:\\datasets\\saved_model\\hrprine\\tmp\\resnet34_pruning.pth')

def set_model_state(model_name,dict_name,new_value):
    model_dict = get_model(model_name, device=device)
    if dict_name in model_dict.keys():
        model_dict[dict_name] = new_value
        torch.save(model_dict, 'D:\\datasets\\saved_model\\hrprine\\tmp\\vgg19_baseline_91.03.pth')
    else:
        print('error')

def get_pruned_lay_list(path):
    pth_list = glob.glob(path)
    for path in pth_list:
        print(path)
        param_dict = torch.load(path)
        for layer_name, prune_list in param_dict['pruned_layer'].items():

            if 'conv15' in layer_name:
                # print(len(prune_list))
                print(prune_list)


def get_kernel_analyse_by_id(pth_name, conv_name, kernel_id):
    state_dict = get_model(pth_name, device=device)['state_dict']
    kernel = state_dict[conv_name][kernel_id]
    return kernel.mean(), kernel.sum(), torch.var(kernel).item(), torch.norm(kernel).item()

def get_kernel_static_graph(file_path, type_idx):
    static_list = []

    note_idx_list = []
    with open(file_path, encoding='UTF-8') as f:
        idx = 0
        for line in f.readlines():

            if 'drop' in line:
                note_idx_list.append(idx)
            else:
                idx+=1
                seprate_list = line.split(',')
                static_list.append(seprate_list[type_idx])
    # x_aixs = [i for i in range(len(mean_list))]
    static_list = list(map(float,static_list))
    x_aixs = np.linspace(1, len(static_list), len(static_list))
    plt.ylim(min(static_list), max(static_list))
    plt.scatter(x_aixs, static_list)
    static_list_np = np.array(static_list)
    note_list = list(static_list_np[note_idx_list])
    plt.scatter(note_idx_list, note_list)
    plt.show()

def fix_shutcut_tensor(shutcut_tensor,original_channel_num):
    if original_channel_num > shutcut_tensor.size()[1]:
        fix_tensor = torch.zeros(shutcut_tensor.size()[0], original_channel_num - shutcut_tensor.size()[1],
                                    shutcut_tensor.size()[2], shutcut_tensor.size()[3])
        shutcut_tensor = torch.cat((shutcut_tensor, fix_tensor.cuda()), 1)
    return shutcut_tensor
if __name__ == '__main__':
    # print("bad")
    # get_pruned_lay_list('D:\\datasets\\saved_model\\hrprine\\bad\\*.pth')
    # print("good")
    # get_pruned_lay_list('D:\\datasets\\saved_model\\hrprine\\good\\*.pth')

    #展示图形
    # get_kernel_static_graph('D:\\15.txt',3)

    # 给保存的pth设置新属性
    # vgg19_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 512]
    # set_model_state('vgg19_baseline.pth', 'cfg', vgg19_cfg)

    # print(get_kernel_analyse_by_id('feature.conv15.weight', 0))

    # model = get_model('vgg_pruning_91.390.pth',device=device)
    # clean_status(model, 'pruned_layer')
    # get_model_statics('vgg_full_baseline.pth')
    # get_model_statics('my_model_pruned.pth')
    # get_prune_model('pruned_0.8_random_25_66.pth', 1, 'o_mean')
    # prune_cfg_list = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, [48, 183, 313, 411], -1]
    # a = get_parameter_by_list('vgg_full_baseline', prune_cfg_list)
    # kernels = a['state_dict']['feature.conv15.weight']
    # for kernel in kernels:
    #     base = kernel.shape[0]*kernel.shape[1]*kernel.shape[2]
    #     fraction = torch.abs(kernel).sum()
    #     print(fraction.item()/base)
    # for parameter in a['state_dict']['feature.conv15.weight']:

    # get_prune_model('vgg_full_baseline.pth', -100, 'o_mean')#获取baseline完整模型数据
    # get_prune_model('pruned_0.8_random_10_61.pth', 1, 'o_mean')
    # get_prune_model('pruned_0.8_random_10_65.pth', 1, 'o_mean')
    # get_prune_model('pruned_0.8_random_12_69.pth', 1, 'o_mean')
    # get_prune_model('pruned_0.8_random_12_81.pth', 1, 'o_mean')
    # get_prune_model('pruned_0.8_random_14_57.pth', 1, 'o_mean')
    # get_prune_model('pruned_0.8_random_19_69.pth', 1, 'o_mean')

    change_model_dict()
