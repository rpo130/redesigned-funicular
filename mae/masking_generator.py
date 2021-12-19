# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import random
import math
import numpy as np
from PIL import Image
from torch import Tensor
import torch
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask # [196]

#黑块嗅探
class BlockDetectGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        #block num in height, block num in width
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self, img : Image.Image):
        img_tensor : Tensor = transforms.ToTensor()(img);
        # print(img_tensor);
        #c h w
        from einops import rearrange
        v_img_block : Tensor = rearrange(img_tensor, 'c (h p1) (w p2) -> (h w) c p1 p2', h=self.height, w=self.width)
        bl = [];
        hitCount = 0;
        for vv in v_img_block:
            hitAvg : bool = False;
            # print(f'vv {vv.shape}')
            for cc in vv:
                #0.5
                if abs(cc.mean() - 0.5) < 0.1 :
                    hitAvg = True;
                #0
                if abs(cc.mean()) < 0.1 :
                    hitAvg = True;

            if hitAvg and hitCount < self.num_mask :
                bl.append(1);
                hitCount += 1;
            else :
                bl.append(0);

        mask = np.array(bl);

        #debug use
        # bool_masked_pos = torch.from_numpy(mask)
        # bool_masked_pos = bool_masked_pos[None, :]
        # bool_masked_pos = bool_masked_pos.to('cpu', non_blocking=True).flatten(1).to(torch.bool)

        # img_squeeze = rearrange(img_tensor[None, :], 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=16, p2=16)
        # img_patch = rearrange(img_squeeze, 'b n p c -> b n (p c)')

        # m = torch.ones_like(img_patch)
        # m[bool_masked_pos] = 0
        # m = rearrange(m, 'b n (p c) -> b n p c', c=3)
        # m = rearrange(m, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=16, p2=16, h=14, w=14)
        # imshow(img_tensor)
        # imshow(m[0])
        return mask # [196]

if __name__ == '__main__' :
    with open(r'D:\OneDrive\研究生\模式识别\大作业\redesigned-funicular\output\mask_img.jpg', 'rb') as f:
        img = Image.open(f)
        img.convert('RGB')
    print(img)
    b = BlockDetectGenerator(14, 0)(img)
    print(b.shape)
    print(b)
