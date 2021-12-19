from einops import rearrange
import numpy as np
import torch
from cnn.masking_generator import RandomMaskingGenerator
import torchvision

def get_masked_pixel(mask_ratio) :
    #生成掩码 #input-size 32 x 32
    #patch-size 4 x 4
    #patch-num 8 x 8
    masked_position_generator = RandomMaskingGenerator((8,8), mask_ratio)
    masked_patch = masked_position_generator();
    masked_pixel = np.zeros((1,3,4,4));
    for i in masked_patch:
        if i == 0:
            masked_pixel = np.vstack((masked_pixel, np.zeros((1,3,4,4,))));
        else:
            masked_pixel = np.vstack((masked_pixel, np.ones((1,3,4,4,))));
    #去除首行
    masked_pixel = masked_pixel[1:]
    #shape 转换
    masked_pixel = torch.from_numpy(masked_pixel)
    masked_pixel = rearrange(masked_pixel, '(n1 n2) c h w-> c (n1 h) (n2 w) ', n1=8, n2=8)
    # imshow(torchvision.utils.make_grid(masked_pixel))
    # print(masked_pixel.shape);

    # imshow(torchvision.utils.make_grid(images[0]), images[0] * masked_pixel)
    # imshow(images[0], images[0] * masked_pixel)
    # c h w
    return masked_pixel;

class ToRamdomMask:
    def __call__(self, pic):
        m = get_masked_pixel();
        r = pic * m;
        r = r.type(torch.float32)
        # print(f'{r.shape} {r.dtype} {pic.dtype}')
        return r;

    def __repr__(self):
        return self.__class__.__name__ + '()'

#see data
#----------
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
#----------

#随机掩盖图像块
class ToRamdomMask:
    def __init__(self, mask_ratio = 0.35) -> None:
        self.m = get_masked_pixel(mask_ratio=mask_ratio);
        self.m = self.m.to(torch.bool);

    def __call__(self, pic : torch.Tensor):
        p = pic.clone();
        p[self.m] = 0
        r = p
        return r;

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
    def refreshM(self, mask_ratio = 0.35) :
        self.m = get_masked_pixel(mask_ratio=mask_ratio);
        self.m = self.m.to(torch.bool)

    def getMaskBlock(self) -> torch.Tensor :
        return self.m
