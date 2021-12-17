from cnn.test import restore_model, test_all_detail_data,test_rand_data,test_all_data
from mae import run_mae_pretraining as rmp

from pathlib import Path
if __name__ == '__main__' : 
    # net = restore_model();
    # test_rand_data(net);
    # test_all_data(net);
    # test_all_detail_data(net);

    #mae pretrain
    opts = rmp.get_args()
    # opts.data_path = '/home/featurize/data'
    # opts.model = 'pretrain_mae_base_patch16_224'
    # opts.opt_betas=[0.9, 0.95]
    # opts.warmup_epochs=40
    # opts.epochs=80
    # opts.output_dir = 'output/pretrain_mae_base_patch16_224'
    # opts.device = "cuda" if torch.cuda.is_available() else 'cpu'
    # opts.num_workers = 5
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    rmp.main(opts)