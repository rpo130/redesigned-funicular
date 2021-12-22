from mae import run_mae_vis
import os
import torch
from PIL import Image
current_dir = os.path.dirname(os.path.realpath(__file__))
pre_dir = os.path.abspath(os.path.join(os.path.realpath(__file__), "../.."))












if __name__=='__main__':
    # img_path = './data/0_29.jpg'
    # img_path = './data/9_9947.jpg'
    # img_path = './data/0_1811.jpg'
    # img_path =  "./data/ILSVRC2012_val_00031649.JPEG"
    # img_path =  "./data/a.jpg"
    # img_path =  "./data/a1.jpg"
    img_path =  "./data/a2.jpg"
    # img_path =  "./data/b.jpg"
    output_dir = current_dir+'/output';
    # model_path = 'D:\OneDrive\研究生\模式识别\大作业\pretrain_mae_vit_base_mask_0.75_400e.pth';
    model_path = 'D:\OneDrive\研究生\模式识别\大作业\cifar10-model\checkpoint-159.pth';
    data_path = 'D:\OneDrive\研究生\模式识别\大作业\src\cnn\data';
    device_type = 'cpu';

    args = run_mae_vis.get_args([img_path, output_dir, model_path,
                                    "--device", device_type,
                                    # "--input_size", '32'
                                    "--mask_ratio", "0.75"
                                    ]);

    args.data_path = data_path

    run_mae_vis.main(args)