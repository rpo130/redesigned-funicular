#load data
#----------
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#----------

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
def imshow(oriimg, img):
    img = img / 2 + 0.5     # unnormalize
    plt.figure()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    oriimg = oriimg / 2 + 0.5     # unnormalize
    plt.figure()
    nporiimg = oriimg.numpy()
    plt.imshow(np.transpose(nporiimg, (1, 2, 0)))
    plt.show()
#----------
# get some random training images
dataiter = iter(testloader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))

from masking_generator import RandomMaskingGenerator
from einops import rearrange
def get_masked_pixel() :
    #生成掩码 #input-size 32 x 32
    #patch-size 4 x 4
    #patch-num 8 x 8
    masked_position_generator = RandomMaskingGenerator((8,8), 0.75)
    masked_patch = masked_position_generator();
    masked_pixel = np.zeros((1,3,4,4));
    for i in masked_patch:
        if i == 1:
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
    return masked_pixel;

#切分小块
# mask = rearrange(mask, 'c (h1 h2) (w1 w2) -> (h1 w1) c h2 w2', h2=4, w2=4)