#load data
#----------
import PIL
import torch
from torch._C import device
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os

import PIL
from PIL import Image

from pathlib import Path

from timm.models import create_model
from mae.transforms import ToTensor

import mae.utils as utils
import mae.modeling_pretrain as modeling_pretrain
from mae.datasets import DataAugmentationForMAE

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import mae.run_mae_vis
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
pre_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), "../.."))

import sys
sys.path.append('.')
from cnn.predict import ToRamdomMask, getImshowData, imshow, imshowPIL

img_path = current_dir + '/mae/files/0_29.jpg'
output_dir = current_dir+'/output';
# model_path = 'D:\OneDrive\研究生\模式识别\大作业\pretrain_mae_vit_base_mask_0.75_400e.pth';
model_path = 'D:\OneDrive\研究生\模式识别\大作业\cifar10-model\checkpoint-159.pth';
data_path = 'D:\OneDrive\研究生\模式识别\大作业\src\cnn\data';
cnn_model_path = 'D:\OneDrive\研究生\模式识别\大作业\src\cnn\model\cifar_net.pth'
device_type = 'cpu';

args = mae.run_mae_vis.get_args([img_path, output_dir, model_path,
                                 "--device", device_type,
                                ]);

args.data_path = data_path

device = torch.device(args.device)
mae_model = mae.run_mae_vis.get_model(args)
mae_model.to(device)
checkpoint = torch.load(args.model_path, map_location=args.device)
mae_model.load_state_dict(checkpoint['model'])
mae_model.eval()

from torchvision import transforms

import matplotlib.pyplot as plt
def squeeze(img, p_size) :
    # plt.subplot(311)
    # plt.imshow(getImshowData(img[0], unnor=True))

    t = None;
    t = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c ', p1=p_size, p2=p_size)

    # plt.subplot(312)
    # plt.imshow(torchvision.utils.make_grid(t[0]))

    # imshow(torchvision.utils.make_grid(t[0]))

    t = rearrange(t, 'b n p c -> b n (p c)')
    # plt.subplot(313)
    # plt.imshow(getImshowData(t[0], unnor=True))
    # imshow(torchvision.utils.make_grid(t[0]))
    # plt.show();
    return t;

def rev_squeeze(img, p_size) :
    t = None;
    t = rearrange(img, 'b n (p c) -> b n p c', c=3)
    t = rearrange(t, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=p_size, p2=p_size, h=14, w=14)
    return t;

def mae_reconstruct(args, model, img : torch.Tensor):
    # img = img.resize((224,224))
    img = transforms.Resize(size=(224,224))(img)
    # imshow(img, unnor=True);
    
    #为了使归一化正常，需要原图进，归一化后再做mask
    #预期图片为 224 x 224
    cudnn.benchmark = True

    patch_size = model.encoder.patch_embed.patch_size
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size
    args.mask_ratio = 0.20
    transforms_mae = DataAugmentationForMAE(args)
    img, bool_masked_pos = transforms_mae(img)
    # imshow(img, unnor=True);

    bool_masked_pos = torch.from_numpy(bool_masked_pos)

    with torch.no_grad():
        img = img[None, :]

        bool_masked_pos = bool_masked_pos[None, :]
        img = img.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)

        # imshow(img[0, :].clip(0, 0.996), unnor=True);
        outputs = model(img, bool_masked_pos)


        #save original img
        mean = torch.as_tensor((0.5,0.5,0.5)).to(device)[None, :, None, None]
        std = torch.as_tensor((0.5,0.5,0.5)).to(device)[None, :, None, None]
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
        img = ToPILImage()(mask[0, :])
        img.save(f"{args.save_path}/mask.jpg")

        pos_img = torch.zeros_like(img_patch);

        # outputs = outputs * 0.5 + 0.5;
        pos_img[bool_masked_pos] = outputs;
        pos_img = rearrange(pos_img, 'b n (p c) -> b n p c', c=3)
        # pos_img = pos_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
        pos_img = rearrange(pos_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)
        img = ToPILImage()(pos_img[0, :])
        img.save(f"{args.save_path}/output_pos.jpg")


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

        ret_img = rec_img[0, :].clip(0,0.996); 

        return ret_img

def mae_reconstruct_call(img : torch.Tensor) :
    return mae_reconstruct(args, mae_model, img);

import cnn.predict as ct
class MaeReconstructTrans:
    def __call__(self, pic):
        recimg = mae_reconstruct_call(pic);
        return recimg;

    def __repr__(self):
        return self.__class__.__name__ + '()'



# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

masked_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ToRamdomMask(),
     transforms.ToPILImage(),
     MaeReconstructTrans(),
     transforms.Resize(size=(32,32)),
    ])

initial_transform = transforms.Compose(
    [
        # transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ])

block_transform = transforms.Compose(
    [
    ToRamdomMask()
    ]
)

mae_construct_transform = transforms.Compose(
    [transforms.ToPILImage(),
     MaeReconstructTrans(),
     transforms.Resize(size=(32,32))]
)

batch_size = 4

#需要修改trans，做不同的数据对比
testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                       download=False, transform=transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#----------


from cnn.predict import Net;
from cnn.predict import test_all_data, test_all_detail_data, test_rand_data;

def get_predict(outputs) :
    _, predicted = torch.max(outputs, 1)
    return predicted;
    
def basic_trans(img : torch.Tensor) -> torch.Tensor :
    c = torch.zeros_like(img);
    for i in range(len(img)):
        c[i] = initial_transform(img[i].clone())
    return c;

def block_trans(img : torch.Tensor) -> tuple :
    c = torch.zeros_like(img);
    m = torch.zeros_like(img, dtype=torch.bool);

    maskG = ToRamdomMask(mask_ratio=0.2)
    for i in range(len(img)):
        c[i] = maskG(img[i]);
        m[i] = maskG.getMaskBlock().clone();
        maskG.refreshM();
        # c[i] = block_transform(img[i])
    return c, m;

def mae_recons(img : torch.Tensor) :
    c = torch.zeros((batch_size,3, 32, 32))
    for i in range(len(img)):
        # img[i].show();
        # imgPIL = transforms.ToPILImage()(c[i].clip(0, 0.996));
        # imshowPIL(img[i])
        maeFix = mae_reconstruct_call(img[i]);
        c[i] = transforms.Resize(size=(32,32))(maeFix);
        # imshow(c[i]);
    return c;

def tensor2PIL(img : torch.Tensor) :
    temp = img.clone();
    c = list();
    for i in range(len(img)):
        c.append(transforms.ToPILImage()(temp[i].clip(0, 0.996)));
    return c;

if __name__ == '__main__':
    cnn_model = Net()
    #load model
    cnn_model.load_state_dict(torch.load(cnn_model_path))
    cnn_model.eval()

    transNum = 4;
    correct = [0 for i in range(transNum)]
    total = [0 for i in range(transNum)]

    dataiter = iter(dataloader);

    printI=0
    for d in dataiter:
        images, labels = d

        #原始
        # imshow(torchvision.utils.make_grid(images))

        #归一化
        images_basic = basic_trans(images);
        # imshow(torchvision.utils.make_grid(images_basic), unnor=True)

        outputs = cnn_model(images_basic)
        predicted = get_predict(outputs);
        print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
        print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))
        total[0] += labels.size(0)
        correct[0] += (predicted == labels).sum().item()

        #mask
        #-----------------------------------------------------------
        images_block, mask_block = block_trans(images_basic);
        #unnormalize 会导致黑块消失
        # imshow(torchvision.utils.make_grid(images_block), unnor=True)

        #classify block
        outputs_block = cnn_model(images_block)
        predicted_block = get_predict(outputs_block);
        print('Predicted2: ', ' '.join('%5s' % classes[predicted_block[j]] for j in range(batch_size)))
        total[1] += labels.size(0)
        correct[1] += (predicted_block == labels).sum().item()
        #-----------------------------------------------------------

        #mae
        #-----------------------------------------------------------
        outputs_mae_construct = mae_recons(images_block)

        outputs_mae = cnn_model(outputs_mae_construct)
        predicted_mae = get_predict(outputs_mae);
        print('Predicted3: ', ' '.join('%5s' % classes[predicted_mae[j]] for j in range(batch_size)))
        total[2] += labels.size(0)
        correct[2] += (predicted_mae == labels).sum().item()
        # imshow(torchvision.utils.make_grid(outputs_mae_construct), unnor=True)
        #-----------------------------------------------------------
        #normal mae
        #-----------------------------------------------------------
        outputs_normal_mae_construct = mae_recons(images_basic)

        outputs_normal_mae = cnn_model(outputs_normal_mae_construct)
        predicted_normal_mae = get_predict(outputs_normal_mae);
        print('Predicted4: ', ' '.join('%5s' % classes[predicted_normal_mae[j]] for j in range(batch_size)))
        total[3] += labels.size(0)
        correct[3] += (predicted_normal_mae == labels).sum().item()
        # imshow(torchvision.utils.make_grid(outputs_normal_mae_construct), unnor=True)
        #-----------------------------------------------------------


        printI+=batch_size;
        if printI % 40 == 0:
            accuracy = [100 * correct[i] / total[i] for i in range(transNum)]
            
            print(f'Accuracy of the network on the pre {printI} test images:')
            print(f'baseline {accuracy[0]}%')
            print(f'block {accuracy[1]}%')
            print(f'mae {accuracy[2]}%')
            print(f'normal mae {accuracy[3]}%')

    # test_rand_data(cnn_model, testloader, args, [block_transform, mae_construct_transform])
    # test_all_data(cnn_model, testloader, args, [block_transform, mae_construct_transform])
