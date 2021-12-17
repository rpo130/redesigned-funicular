#load data
#----------
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

from PIL import Image

from pathlib import Path

from timm.models import create_model

import utils
import modeling_pretrain
from datasets import DataAugmentationForMAE

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import run_mae_vis
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
pre_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), "../.."))

import sys
sys.path.append('.')
from cnn.test import ToRamdomMask

def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    # parser.add_argument('img_path', type=str, help='input image path')
    # parser.add_argument('save_path', default='./output', type=str, help='save image path')
    # parser.add_argument('model_path', default='D:\OneDrive\研究生\模式识别\大作业\pretrain_mae_vit_base_mask_0.75_400e.pth', type=str, help='checkpoint path of model')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    args = parser.parse_args()
    # args.save_path = './output'
    # args.model_path = 'D:\OneDrive\研究生\模式识别\大作业\pretrain_mae_vit_base_mask_0.75_400e.pth'

    return args;

args = get_args();
args.save_path = current_dir+'/output';
args.model_path = 'D:\OneDrive\研究生\模式识别\大作业\pretrain_mae_vit_base_mask_0.75_400e.pth';
model = run_mae_vis.get_model(args)

def modelEval(model, img):
    args = get_args()

    # args.img_path = '';
    args.save_path = current_dir+'/output';
    args.model_path = 'D:\OneDrive\研究生\模式识别\大作业\pretrain_mae_vit_base_mask_0.75_400e.pth';

    device = torch.device(args.device)
    cudnn.benchmark = True

    # model = run_mae_vis.get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transforms = DataAugmentationForMAE(args)
    img, bool_masked_pos = transforms(img)
    bool_masked_pos = torch.from_numpy(bool_masked_pos)

    with torch.no_grad():
        img = img[None, :]
        bool_masked_pos = bool_masked_pos[None, :]
        img = img.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        outputs = model(img, bool_masked_pos)

        #save original img
        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None]
        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None]
        ori_img = img * std + mean  # in [0, 1]
        img = ToPILImage()(ori_img[0, :])
        img.save(f"{args.save_path}/ori_img.jpg")

        img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[0])
        img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        img_patch[bool_masked_pos] = outputs

        #make mask
        mask = torch.ones_like(img_patch)
        mask[bool_masked_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)

        #save reconstruction img
        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
        rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)
        img = ToPILImage()(rec_img[0, :].clip(0,0.996))
        img.save(f"{args.save_path}/rec_img.jpg")

        #save random mask img
        img_mask = rec_img * mask
        img = ToPILImage()(img_mask[0, :])
        img.save(f"{args.save_path}/mask_img.jpg")
        return rec_img[0, :].clip(0,0.996)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


masked_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ToRamdomMask()
    ])

batch_size = 1

#需要修改trans，做不同的数据对比
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root=pre_dir+'/cnn/data', train=False,
                                       download=False, transform=masked_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#----------


#set up network
#----------
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
#----------

PATH = pre_dir+'/cnn/model/cifar_net.pth'

net = Net()
#load model
net.load_state_dict(torch.load(PATH))

from cnn.test import test_all_data;
#整体数据验证
#----
test_all_data(net)
#----

#分类数据验证
#----
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
#--

i = 0
# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        #----
        pic = images[0]
        recimg = modelEval(model, transforms.ToPILImage()(pic));
        recimg = transforms.Resize(size=(32,32))(recimg)
        images[0] = recimg;
        #----
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

        i = i + 1;
        if i == 200 :
            break;


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    if total_pred[classname] == 0:
        print('skip')
        continue;
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))
#----
