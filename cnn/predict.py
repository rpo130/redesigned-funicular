#load data
#----------
import PIL
from PIL.Image import init
import torch
from torch._C import import_ir_module
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToPILImage
from cnn.ysl_util import get_masked_pixel, ToRamdomMask

import os
current_dir = os.path.dirname(os.path.realpath(__file__))
pre_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), "../.."))

def get_dataloader(data_path) :
    #需要修改trans，做不同的数据对比
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=False, transform=transform)
    # testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
    #                                     download=False, transform=masked_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)
    return testloader;


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

masked_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     ToRamdomMask()
    ])

batch_size = 4

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#see data
#----------
import matplotlib.pyplot as plt
import numpy as np

def getImshowData(img : torch.Tensor, unnor = False):
    if unnor :
        img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0));
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show()

# functions to show an image
def imshow(img : torch.Tensor, unnor = False):
    if unnor :
        img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imshowPIL(img : PIL.Image.Image):
    plt.imshow(img)
    plt.show()
#----------


from cnn.net_model import Net;

def restore_model() :
    PATH = pre_dir + '/data/cifar10_classify_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))
    return net;

def test_rand_data(net, dataloader, args, compare_trans = []) :
    dataiter = iter(dataloader if dataloader is not None else get_dataloader(args.data_path))
    images, labels = dataiter.next()
    # ToPILImage()(images[0]).save('./a.jpg')

    # print images
    imshow(torchvision.utils.make_grid(images), unnor=True)
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
    #----------

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                for j in range(4)))

    c_images = images;
    if len(compare_trans) > 0 :
        for i in range(len(c_images)):
            c_images[i] = compare_trans[0](c_images[i])
        imshow(torchvision.utils.make_grid(c_images))

        outputs = net(c_images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted1: ', ' '.join('%5s' % classes[predicted[j]]
                                    for j in range(4)))

    if len(compare_trans) > 1 :
        for i in range(len(c_images)):
            c_images[i] = compare_trans[1](c_images[i])
        imshow(torchvision.utils.make_grid(c_images))

        outputs = net(c_images)
        _, predicted = torch.max(outputs, 1)
        print('Predicted2: ', ' '.join('%5s' % classes[predicted[j]]
                                    for j in range(4)))


def test_all_data(net, dataloader, args, compare_trans = []) :
    #整体数据验证
    #----
    compareLen = len(compare_trans) if compare_trans is not None else 0;

    correct = [0 for i in range(compareLen+1)]
    total = [0 for i in range(compareLen+1)]
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        dl = dataloader if dataloader is not None else get_dataloader(args.data_path)

        printI=0
        for data in dl:
            images, labels = data

            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total[0] += labels.size(0)
            correct[0] += (predicted == labels).sum().item()

            c_images = images;
            for i in range(compareLen) :
                for im in range(len(c_images)):
                    c_images[im] = compare_trans[i](c_images[im]);
                #back to batch
                # calculate outputs by running images through the network
                outputs = net(c_images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total[i+1] += labels.size(0)
                correct[i+1] += (predicted == labels).sum().item()
                
            printI+=batch_size;
            if needPrint(printI):
                accuracy = [100 * correct[i] / total[i] for i in range(compareLen+1)]
                
                print(f'Accuracy of the network on the pre {printI} test images:')
                print(f'baseline {accuracy[0]}%')
                if compareLen >= 1:
                    print(f'block {accuracy[1]}%')
                if compareLen >= 2:
                    print(f'mae {accuracy[2]}%')

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct[0] / total[0]))
    return correct[0] / total[0]

def needPrint(i) :
    # if i % 40 == 0:
    #     return True;
    return False;

def test_all_detail_data(net, dataloader):
    #分类数据验证
    #----
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        dl = dataloader if dataloader is not None else get_dataloader();
        for data in dl:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            print('skip')
            continue;
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))