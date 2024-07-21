import os, json
import torch
from torch import nn
import numpy as np
from modules.tester import Tester
from models.blip import PromptCLS
from dataset import create_dataset 
from dataset import create_sampler 
from dataset import create_loader 
from modules import utils

from modules.config_loader import load_config

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
    print('loading data from %s'%args.ann_path)
    train_dataset, val_dataset, test_dataset = create_dataset('generation_%s'%args.dataset_name, args)
    print('number of training samples: %d'%len(train_dataset))
    print('number of validation samples: %d'%len(val_dataset))
    print('number of testing samples: %d'%len(test_dataset))

    # distribution of diseases
    base_probs = args.distribution
    # normalize
    base_probs = np.array(base_probs) / np.max(base_probs)
    # add extra probs for 4 auxiliry diseases
    base_probs = np.append(base_probs, [1,1,1,1])
    samplers = [None, None, None]

    train_dataloader, val_dataloader, test_dataloader = create_loader([train_dataset, val_dataset, test_dataset], samplers, batch_size=[args.batch_size]*3, num_workers=[4,4,4], is_trains=[True, False, False], collate_fns=[None, None, None]) 

    model = PromptCLS(args)
    args.load_pretrained = '/root/OriginPromptMRG/PromptMRG/results/model_best.pth'
    if args.load_pretrained:
        state_dict = torch.load(args.load_pretrained, map_location="cpu")
        msg = model.load_state_dict(state_dict, strict=False)
        print("load checkpoint from {}".format(args.load_pretrained))

    # get function handles of loss and metrics
    # criterion_cls = nn.CrossEntropyLoss()


    model = model.to(device)   
    # build trainer and start to train
    tester = Tester(model, args,  test_dataloader, device)
    tester.test_blip()

if __name__ == '__main__':
    main(config = 'configs/PromptMRGCLS_V2.yaml',stage='full')
