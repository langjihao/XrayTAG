import os, json
import torch
from torch import nn
import numpy as np
from modules.trainer import Trainer
from models.Chexfusion import Chexfusion
from dataset import create_dataset 
from dataset import create_sampler 
from dataset import create_loader 
from modules import utils
from modules.loss import get_loss

from modules.config import *

os.environ['TOKENIZERS_PARALLELISM'] = 'True'


def main(config, stage='dev'):
    '''
    config :path to the config file
    stage : str, one of dev, exp, full
    '''
    # parse arguments
    args = load_config(config)
    args.stage = stage
    device = torch.device(args.device)

    # fix random seeds
    seed = args.seed + utils.get_rank() # from blip
    torch.manual_seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


    #### Dataset #### 
        # 不同阶段加载不同的数据集
    if stage == 'dev':
        args.ann_path = '/hy-tmp/files256/mimic_dev.json'
    elif stage == 'exp':
        args.ann_path = '/hy-tmp/files256/mimic_exp.json'
    elif stage == 'full':
        args.ann_path = '/hy-tmp/files256/mimic_full.json'
    else:
        raise ValueError('stage should be one of dev, exp, full')
    train_dataset, val_dataset, test_dataset = create_dataset('generation_%s'%args.dataset_name, args)
   
    print(f'Number of training samples: {len(train_dataset)}')
    print(f'Number of validation samples: {len(val_dataset)}')
    print(f'Number of testing samples: {len(test_dataset)}')

    samplers = [None, None, None]

    train_dataloader, val_dataloader, test_dataloader = create_loader([train_dataset, val_dataset, test_dataset], samplers, batch_size=[args.batch_size]*3, num_workers=[4,4,4], is_trains=[True, False, False], collate_fns=[None, None, None]) 

    model = Chexfusion(args)
    if args.load_pretrained:
        state_dict = torch.load(args.load_pretrained, map_location="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        print("load checkpoint from {}".format(args.load_pretrained))

    # get function handles of loss and metrics
    criterion_cls = get_loss(type=args.loss,class_instance_nums=args.class_instance_nums,total_instance_num=args.total_instance_num)


    model = model.to(device)   
    # build trainer and start to train
    trainer = Trainer(model, criterion_cls, args, train_dataloader, val_dataloader, test_dataloader, device)
    trainer.train()

if __name__ == '__main__':
    main(config = './configs/Chexfusion.yaml',stage='full')
