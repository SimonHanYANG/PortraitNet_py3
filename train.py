import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.nn import DataParallel

import numpy as np
import argparse
import time
import os
import shutil
from easydict import EasyDict as edict
from yaml import load

from dataset import Human
from data_aug import Normalize_Img, Anti_Normalize_Img
from focal_loss import FocalLoss

def calcIOU(img, mask):
    sum1 = img + mask
    sum1[sum1>0] = 1
    sum2 = img + mask
    sum2[sum2<2] = 0
    sum2[sum2>=2] = 1
    if np.sum(sum1) == 0:
        return 1
    else:
        return 1.0*np.sum(sum2)/np.sum(sum1)
    
def get_parameters(model, args, useDeconvGroup=True):
    lr_0 = []
    lr_1 = []
    params_dict = dict(model.named_parameters())
    for key, value in params_dict.items():
        if 'deconv' in key and useDeconvGroup==True:
            print ("useDeconvGroup=True, lr=0, key: ", key)
            lr_0.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_0, 'lr': args.lr * 0},
              {'params': lr_1, 'lr': args.lr * 1}]
    return params, [0., 1.]

def adjust_learning_rate(optimizer, epoch, args, multiple):
    """Sets the learning rate to the initial LR decayed by 0.95 every 20 epochs"""
    # lr = args.lr * (0.95 ** (epoch // 4))
    lr = args.lr * (0.95 ** (epoch // 20))
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr * multiple[i]
    pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    pass

def loss_KL(student_outputs, teacher_outputs, T):
    """
    Code referenced from: 
    https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
    
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1), 
                             F.softmax(teacher_outputs/T, dim=1)) * T * T
    return KD_loss

def train(dataLoader, netmodel, optimizer, epoch, exp_args):
    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    
    losses = AverageMeter('losses')
    losses_mask = AverageMeter('losses_mask')
    
    if exp_args.addEdge == True:
        losses_edge_ori = AverageMeter('losses_edge_ori')
        losses_edge = AverageMeter('losses_edge')
    
    if exp_args.stability == True:
        losses_mask_ori = AverageMeter('losses_mask_ori')
        losses_stability_mask = AverageMeter('losses_stability_mask')
        losses_stability_edge = AverageMeter('losses_stability_edge')

    netmodel.train() # switch to train mode
    
    loss_Softmax = nn.CrossEntropyLoss(ignore_index=255) # mask loss
    # in our experiments, focalloss is better than l2 loss
    loss_Focalloss = FocalLoss(gamma=2) # boundary loss
    loss_l2 = nn.MSELoss() # boundary loss
    
    end = time.time()
    for i, (input_ori, input, edge, mask) in enumerate(dataLoader):
        data_time.update(time.time() - end)
        # torch.Size([bs, 3, 224, 224])
        # print("input: ", input.shape)
        # torch.Size([bs, 224, 224])
        # print("edge: ", edge.shape)
        # torch.Size([bs, 224, 224])
        # print("mask: ", mask.shape)
        input_ori_var = Variable(input_ori.cuda())
        input_var = Variable(input.cuda())
        edge_var = Variable(edge.cuda())
        mask_var = Variable(mask.cuda())
        
        # whether to add boundary auxiliary loss 
        if exp_args.addEdge == True:
            output_mask, output_edge = netmodel(input_var)
            # ([4, 2, 224, 224])
            print("Output mask shape: ", output_mask.shape)
            # ([4, 2, 224, 224])
            print("Output edge shape: ", output_edge.shape)
            
        
        

def save_checkpoint(state, is_best, root, filename='checkpoint.pth.tar'):
    torch.save(state, root+filename)
    if is_best:
        shutil.copyfile(root+filename, root+'model_best.pth.tar')
       

def main(args):
    cudnn.benchmark = True
    assert args.model in ['PortraitNet', 'ENet', 'BiSeNet'], 'Error!, <model> should in [PortraitNet, ENet, BiSeNet]'
    
    config_path = args.config_path
    print ('===========> loading config <============')
    print ("config path: ", config_path)
    with open(config_path,'rb') as f:
        cont = f.read()
    
    from yaml import FullLoader
    cf = load(cont, Loader = FullLoader)
    
    print ('===========> loading data <===========')
    exp_args = edict()
    
    exp_args.istrain = cf['istrain'] # set the mode 
    exp_args.task = cf['task'] # only support 'seg' now
    exp_args.datasetlist = cf['datasetlist']
    exp_args.model_root = cf['model_root'] 
    exp_args.data_root = cf['data_root']
    exp_args.file_root = cf['file_root']
    
    # the height of input images, default=224
    exp_args.input_height = cf['input_height']
    # the width of input images, default=224
    exp_args.input_width = cf['input_width']
    
    # if exp_args.video=True, add prior channel for input images, default=False
    exp_args.video = cf['video']
    # the probability to set empty prior channel, default=0.5
    exp_args.prior_prob = cf['prior_prob']
    
    # whether to add boundary auxiliary loss, default=False
    exp_args.addEdge = cf['addEdge']
    # the weight of boundary auxiliary loss, default=0.1
    exp_args.edgeRatio = cf['edgeRatio']
    # whether to add consistency constraint loss, default=False
    exp_args.stability = cf['stability']
    # whether to use KL loss in consistency constraint loss, default=True
    exp_args.use_kl = cf['use_kl']
    # temperature in consistency constraint loss, default=1
    exp_args.temperature = cf['temperature'] 
    # the weight of consistency constraint loss, default=2
    exp_args.alpha = cf['alpha'] 
    
    # input normalization parameters
    exp_args.padding_color = cf['padding_color']
    exp_args.img_scale = cf['img_scale']
    # BGR order, image mean, default=[103.94, 116.78, 123.68]
    exp_args.img_mean = cf['img_mean']
    # BGR order, image val, default=[1/0.017, 1/0.017, 1/0.017]
    exp_args.img_val = cf['img_val'] 
    
    # whether to use pretian model to init portraitnet
    exp_args.init = cf['init'] 
    # whether to continue training
    exp_args.resume = cf['resume'] 
    
    # if exp_args.useUpsample==True, use nn.Upsample in decoder, else use nn.ConvTranspose2d
    exp_args.useUpsample = cf['useUpsample'] 
    # if exp_args.useDeconvGroup==True, set groups=input_channel in nn.ConvTranspose2d
    exp_args.useDeconvGroup = cf['useDeconvGroup'] 
    
    # set training dataset
    exp_args.istrain = True
    
    dataset_train = Human(exp_args)
    print ("image number in training: ", len(dataset_train))
    dataLoader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batchsize, 
                                                   shuffle=True, num_workers= args.workers)
    
    # set testing dataset
    exp_args.istrain = False
    dataset_test = Human(exp_args)
    print ("image number in testing: ", len(dataset_test))
    dataLoader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, 
                                                  shuffle=False, num_workers=args.workers)
    
    print ("finish load dataset ...")
    
    model_chosen = False
    if args.model == 'PortraitNet':
        # train our model: portraitnet
        # import model_mobilenetv2_seg_small as modellib
        # import model.model_mobilenetv2_seg_small as modellib
        # netmodel = modellib.MobileNetV2(n_class=2, 
        #                                 useUpsample=exp_args.useUpsample, 
        #                                 useDeconvGroup=exp_args.useDeconvGroup, 
        #                                 addEdge=exp_args.addEdge, 
        #                                 channelRatio=1.0, 
        #                                 minChannel=16, 
        #                                 weightInit=True,
        #                                 video=exp_args.video).cuda()
        
        # myModel
        from model import MobileNetV2
        netmodel = MobileNetV2(n_class=2, 
                            useUpsample=exp_args.useUpsample, 
                            useDeconvGroup=exp_args.useDeconvGroup, 
                            addEdge=exp_args.addEdge, 
                            channelRatio=1.0, 
                            minChannel=16, 
                            video=exp_args.video).cuda()
        print(netmodel)
        model_chosen = True
        print ("finish load PortraitNet ...")
    else:
        print("Error Chosen Network Model...")
    
    # optimizer = torch.optim.SGD(netmodel.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weightdecay)  
    params, multiple = get_parameters(netmodel, args, useDeconvGroup=exp_args.useDeconvGroup)
    # optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weightdecay) 
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weightdecay) 
    
    # donot use pretrained model
    if exp_args.init:
        print("Cannot use Mobilev2 pretrained model....")

    minLoss = 10000
    gap = 0
    
    # Start training
    if model_chosen:
        for epoch in range(gap, 2000):
            adjust_learning_rate(optimizer, epoch, args, multiple)
            print ('===========>   training    <===========')
            train(dataLoader_train, netmodel, optimizer, epoch, exp_args)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training code')
    parser.add_argument('--model', default='PortraitNet', type=str, 
                        help='<model> should in [PortraitNet, ENet, BiSeNet]')
    parser.add_argument('--config_path', 
                        default="/home/simon/SimonWorkspace/PortraitNet_py3/config/model_mobilenetv2_with_two_auxiliary_losses.yaml",
                        # default="/home/simon/SimonWorkspace/PortraitNet_py3/config/super_model_mobilenetv2_with_two_auxiliary_losses.yaml",
                        # default='/home/dongx12/PortraitNet/config/model_mobilenetv2_without_auxiliary_losses.yaml', 
                        type=str, help='the config path of the model')
    
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    # for real train: batchsize=64
    parser.add_argument('--batchsize', default=4, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weightdecay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--printfreq', default=100, type=int, help='print frequency')
    parser.add_argument('--savefreq', default=1000, type=int, help='save frequency')
    parser.add_argument('--resume', default=False, type=bool, help='resume')
    args = parser.parse_args()
    
    main(args)