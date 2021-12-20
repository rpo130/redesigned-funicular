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
from torchvision import transforms
from mae.masking_generator import imshow

import mae.utils as utils
import mae.modeling_pretrain as modeling_pretrain
from mae.datasets import DataAugmentationForMAE

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_STD
from .run_mae_vis import get_model

def rebuild_pic_single(args, model, img):
    device = torch.device(args.device)
    patch_size = model.encoder.patch_embed.patch_size
    # print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    tran_mae = DataAugmentationForMAE(args)

    img, bool_masked_pos = tran_mae(img)
    bool_masked_pos = torch.from_numpy(bool_masked_pos)

    with torch.no_grad():
        img = img[None, :]
        bool_masked_pos = bool_masked_pos[None, :]
        img = img.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        outputs = model(img, bool_masked_pos)

        DATA_MEAN = (0.5,0.5,0.5)
        DATA_STD = (0.5,0.5,0.5)

        #save original img
        #---------------------------
        mean = torch.as_tensor(DATA_MEAN).to(device)[None, :, None, None]
        std = torch.as_tensor(DATA_STD).to(device)[None, :, None, None]
        ori_img = img * std + mean  # in [0, 1]
        # print(f'unnorm : {img.mean()}, {img.var()}')
        img = ToPILImage()(ori_img[0, :])
        img.save(f"{args.save_path}/ori_img.jpg")
        #---------------------------

        #记录原始模型输入
        #逆归一化之后
        #-------------------------------------
        old_input_img = ori_img;
        # print(f'norm : {old_input_img.mean()}, {old_input_img.var()}')

        old_input_img_squeeze = rearrange(old_input_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[0])
        #normal
        old_input_img_squeeze_normal = (old_input_img_squeeze - old_input_img_squeeze.mean(dim=-2, keepdim=True)) / (old_input_img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)

        # print(f'norm aft conv1 data , mean-2 : {old_input_img.mean()}, diy : {old_input_img.sum()}, {old_input_img.shape}')
        old_input_img_squeeze_fill = old_input_img_squeeze.clone();

        #cal squeeze by considering before block
        #-------------------------------------------------
        lastNotBlockArea = None;
        hit = 0;
        for bi in range(len(old_input_img_squeeze_fill)) :
            mask_batch = bool_masked_pos[bi];
            for blocki in range(len(old_input_img_squeeze_fill[bi])) :
                currBlock = old_input_img_squeeze_fill[bi][blocki];
                mask_flag = mask_batch[blocki];
                if mask_flag :
                    if lastNotBlockArea != None :
                        old_input_img_squeeze_fill[bi][blocki] = lastNotBlockArea;
                        pass
                else :
                    pass;

                hit += 1;
                if lastNotBlockArea is None :
                    lastNotBlockArea = currBlock;
                else :
                    old_w = int(hit * 0.4);
                    new_w = hit - old_w
                    lastNotBlockArea = ( old_w * lastNotBlockArea + new_w * currBlock ) / hit;
            lastNotBlockArea = None;
            hit = 0;
        #-------------------------------------------------

        #aft normal
        old_input_img_squeeze_normal_squeeze = rearrange(old_input_img_squeeze_normal, 'b n p c -> b n (p c)')

        old_input_img_squeeze_normal_squeeze[bool_masked_pos] = outputs

        old_input_rec_img = rearrange(old_input_img_squeeze_normal_squeeze, 'b n (p c) -> b n p c', c=3)
        old_input_rec_img = old_input_rec_img  * (old_input_img_squeeze_fill.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + old_input_img_squeeze_fill.mean(dim=-2, keepdim=True)

        old_input_rec_img = rearrange(old_input_rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)

        # imshow(old_input_rec_img[0].clip(0,1))
        #-------------------------------------

        img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size[0], p2=patch_size[0])
        # img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
        img_norm = img_squeeze
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')

        #TODO need to construct using the same distribution but block are can't infer
        # img_patch[bool_masked_pos] = outputs

        #make mask
        mask = torch.ones_like(img_patch)
        mask[bool_masked_pos] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)

        anti_mask = torch.zeros_like(img_patch)
        anti_mask[bool_masked_pos] = 1
        anti_mask = rearrange(anti_mask, 'b n (p c) -> b n p c', c=3)
        anti_mask = rearrange(anti_mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)

        #save reconstruction img
        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        # rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2, keepdim=True)
        rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=14, w=14)
        #only change predict area
        rec_img[anti_mask.to(torch.bool)] = old_input_rec_img[anti_mask.to(torch.bool)]
        # img = ToPILImage()(rec_img[0, :].clip(0,0.996))
        # img.save(f"{args.save_path}/rec_img.jpg")

        #normalize
        rec_img = rec_img[0];
        rec_img = transforms.Normalize(DATA_MEAN, DATA_STD)(rec_img)
        return rec_img;