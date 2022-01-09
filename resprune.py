import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import *

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=164,
                    help='depth of the resnet')
parser.add_argument('--percent', type=float, default=0.5,
                    help='scale sparse rate (default: 0.5)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = resnet(depth=args.depth, dataset=args.dataset)

if args.cuda:
    print('prune here cuda')
    model.cuda()
    device = "cuda"
else:
    device = "cpu"

if args.model:
    # checkpoint具体的样子
    # save_checkpoint({
    #     'epoch': epoch + 1,
    #     'state_dict': model.state_dict(),
    #     'best_prec1': best_prec1,
    #     'optimizer': optimizer.state_dict(),
    # }, is_best, filepath=args.save)
    # 由is_best在save_checkpoint函数中控制,确保model是最佳model
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

total = 0

# 确定好到底有多少channel是属于batchnorm的
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        # 对于batchnorm2d这个module,m.weight.shape就是channel的个数
        # 所以这里第一个m.weight.shape=m.bias.shape=torch.Size([16])=m.weight.data.shape[0]
        # print('batchnorm module\'s weight shape: ', m.weight.shape)
        # print('batchnorm module\'s bias shape: ', m.bias.shape)
        # 对于batchnorm, gamma*x+beta中的gamma在pytorch中就是weight, beta则为bias
        # 所以此处m.weight中的weight即充当gamma的角色
        # total：是模型中总共batchnorm的channel个数
        total += m.weight.data.shape[0]

# 将每一层属于batchnorm的gamma值都提取出来
bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index + size)] = m.weight.data.abs().clone()
        index += size

# 按照想保留的百分比, 截取出想保留的channel
# 从小到大排列
y, i = torch.sort(bn)  # y, i: sorted bn ——> y: sorted weight, i: corresponding index
# Eg:
# bn = torch.Tensor([1, 5, 6, 2, 7, 67, 8, 9, 3, 0])
# y, i = torch.sort(bn)
# y: tensor([0, 1, 2, 3, 5, 6, 7, 8, 9, 67])
# i: tensor([9, 0, 3, 8, 1, 2, 4, 6, 7, 5])
thre_index = int(total * args.percent)
# todo(只是为了这个色): to cuda. 这里的.cuda()是必要的,否则会出现错误.原文件中没有这个,依据版本,可能要自己加上
# todo(只是为了这个色): RuntimeError: Expected object of backend CUDA but got backend CPU for argument #2 'other'
# 找到threshold
thre = y[thre_index].cuda()

pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        # 获取当前channel的weight(gamma)
        weight_copy = m.weight.data.abs().clone()
        # mask的作用：在于把当前的m.weight中大于阈值的挑出来(设置成1，则小于阈值的为0，形成mask)
        # print('mask: ', weight_copy.gt(thre).float())
        mask = weight_copy.gt(thre).float().cuda()
        # pruned：代表总共被prune的channel的个数
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        # 保留>thre的weight与bias的值，<=的全部置零
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        # 记录当前batchnorm层一共保留了几层channel
        cfg.append(int(torch.sum(mask)))
        # 记录当前batchnrom层的mask
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
              format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned / total

print('Pre-processing Successful!')


# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                # rgb   # https://www.programcreek.com/python/example/104838/torchvision.transforms.RandomCrop
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    # .2f, float(correct)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
        float(correct), len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return float(correct) / float(len(test_loader.dataset))


acc = test(model)

print("Cfg:")
print(cfg)
# cfg调控channel个数：
# Cfg:
# [5, 11, 13, 8, 11, 9, 26, 32, 32, 9, 30, 32, 88, 64, 64, 33, 64, 64, 220]


# resnet仅生层network。因加入cfg，所以生成新的网络，主要由cfg调控channel个数
# 此时newmodel即为压缩过后的network
newmodel = resnet(depth=args.depth, dataset=args.dataset, cfg=cfg)
if args.cuda:
    newmodel.cuda()

# 对于此句的解释 (简单解释一下，对核心内容没有影响)：
# param是每层的parameter,
# 长这样： tensor([[[a,b,c], [d,e,f], [g,h,i]], [[], [], []], [[], [], []]], device='cuda:0', requires_grad=True)
# shape: 就第一层而言：torch.Size([16, 3, 3, 3]),代表：(output_size, input_size // group, *kernel_size)
#        其中input_size//group=3, 表示输入图像为3channel,output_size=16为输出channel为16,*kernel_size=(3,3)表示是3x3的kernel
# nelement: 432=16x3x3x3
#
# E.g:
# for param in newmodel.parameters():
#     print('param: ', param)
#     print('param shape: ', param.shape)
#     print('param.nelement: ', param.nelement())
#     break
#
# 如果要获得每层的名字以及parameters的话,可以用：named_parameters()
# E.g:
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)
num_parameters = sum([param.nelement() for param in newmodel.parameters()])

savepath = os.path.join(args.save, "prune.txt")
with open(savepath, "w") as fp:
    fp.write("Configuration: \n" + str(cfg) + "\n")
    fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
    fp.write("Test accuracy: \n" + str(acc))

# 要开始生成真正新的model了
old_modules = list(model.modules())

new_modules = list(newmodel.modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)  # mask before prune at layer batchnorm
# now end_mask:  tensor([0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.], device='cuda:0')
end_mask = cfg_mask[layer_id_in_cfg]  # mask after prune at layer batchnorm
conv_count = 0
print('cfg mask 0: ', end_mask)

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d):
        # get the mask: (arrray([0, 1, 1, 0, 0, 1, ..], dtype=int) of channel after pruning after batchnorm
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        # make sure idx1 is a real numpy list (when size==1, idx1 is a number not a list, we need to change it to list)
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))

        if isinstance(old_modules[layer_id + 1], channel_selection):
            # If the next layer is the channel selection layer,
            # then the current batchnorm 2d layer won't be pruned.
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

            # We need to set the channel selection layer.
            # indexes is a self-defined parameter which plays a role in channel_selection
            # to help to select channels.
            m2 = new_modules[layer_id + 1]  # 此时，m2本质上是channel_selection layer，
            m2.indexes.data.zero_()         # 其含有indexes参数
            m2.indexes.data[idx1.tolist()] = 1.0

            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
        else:
            # This means we need to prune some channels
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        # 开篇第一个conv
        if conv_count == 0:
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        if isinstance(old_modules[layer_id - 1], channel_selection) or \
                isinstance(old_modules[layer_id - 1], nn.BatchNorm2d):
            # This covers the convolutions in the residual block.
            # The convolutions are either after the channel selection layer or
            #                             after the batch normalization layer.
            conv_count += 1
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            # Every conv would be changed it's input channel number
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()     # [output_channel, input_channel, *kernel_size]

            # If it's the last conv in one block (there will be a shortcut[pixelwise sum]),
            # this conv should be changed it's output channel number
            #
            # If the current convolution is not the last convolution in the residual block,
            # then we can change the number of output channels.
            # Currently we use `conv_count` to detect whether it is such convolution.
            if conv_count % 3 != 1:
                # To conv, the shape of weight is: [output_channel, input_channel, *kernel_size]
                w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            continue

        # We need to consider the case where there are downsampling convolutions.
        # For these convolutions, we just copy the weights.
        m1.weight.data = m0.weight.data.clone()
    elif isinstance(m0, nn.Linear):
        # 最后一层FC
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        # m0.weight.data.shape： torch.Size([10, 256]) --> [output_size, input_size]
        # m0.bias.data.shape： torch.Size([10]) --> [output_size]
        # That's why m0.bias.data.clone() is enough. (No need to add [] after data)
        m1.weight.data = m0.weight.data[:, idx0].clone()
        m1.bias.data = m0.bias.data.clone()

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, 'pruned.pth.tar'))

# print(newmodel)
model = newmodel
test(model)
