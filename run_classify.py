from cnn.predict import get_dataloader, restore_model, test_all_data, ToRamdomMask, test_rand_data
import torchvision
import torch
import trans

DATA_MEAN = (0.5, 0.5, 0.5);
DATA_STD = (0.5, 0.5, 0.5)
MASK_PERCENT = 0.1;
EARSE_NUM = [0, 0, 0]



if __name__=='__main__':
    net = restore_model();
    net.eval()
    DATA_PATH = 'D:\OneDrive\研究生\模式识别\大作业\src\cnn\data';
    BATCH_SIZE = 4;

    data_trans = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(DATA_MEAN, DATA_STD)
        ])


    data_masked_trans = torchvision.transforms.Compose(
        [
            data_trans,
            trans.RandomErasing(probability=1,sl=MASK_PERCENT,sh=MASK_PERCENT,r1=1, mean=EARSE_NUM)
        ])

    #baseline
    #-----------------------------
    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                           download=False, transform=data_trans)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    # test_all_data(net, testloader, None);
    #-----------------------------


    acc = [];
    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] :
        data_masked_trans = torchvision.transforms.Compose(
        [
            data_trans,
            trans.RandomErasing(probability=1,sl=i,sh=i,r1=1, mean=EARSE_NUM)
        ])
        print('-----------------------------split------------------------------------')
        print(f'start {i} mask rate')

        sum_ac = 0;
        roun = 5;
        for ii in range(roun):
            #target task
            #-----------------------------
            testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                                download=False, transform=data_masked_trans)

            testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            ac = test_all_data(net, testloader, None);
            sum_ac += ac;
            # test_rand_data(net, testloader, None)
            #-----------------------------
        acc.append(sum_ac / roun)
    print(f'finis run')
    for a in acc:
        print(f'accrate : {a}')
