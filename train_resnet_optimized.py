import os, json
import torch
from torch import nn
import numpy as np
from modules.trainer import Trainer
from models.ResNetOptimized import ResNetOptimized, ResNetBaseline
from dataset import create_dataset 
from dataset import create_sampler 
from dataset import create_loader 
from modules import utils
from modules.loss import get_loss

from modules.config import *

os.environ['TOKENIZERS_PARALLELISM'] = 'True'


def main(config = 'configs/resnet_optimized.yaml', stage='full', model_type='optimized'):
    '''
    config: path to the config file
    stage: str, one of dev, exp, full
    model_type: str, one of optimized, baseline
    '''
    # parse arguments
    args = load_config(config)
    args.stage = stage
    device = torch.device(args.device)

    # fix random seeds
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"使用配置: {config}")
    print(f"训练阶段: {stage}")
    print(f"模型类型: {model_type}")
    print(f"损失函数: {args.loss}")

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
   
    print(f'训练样本数量: {len(train_dataset)}')
    print(f'验证样本数量: {len(val_dataset)}')
    print(f'测试样本数量: {len(test_dataset)}')

    samplers = [None, None, None]
    train_dataloader, val_dataloader, test_dataloader = create_loader(
        [train_dataset, val_dataset, test_dataset], 
        samplers, 
        batch_size=[args.batch_size]*3, 
        num_workers=[4,4,4], 
        is_trains=[True, False, False], 
        collate_fns=[None, None, None]
    ) 

    # 创建模型
    if model_type == 'optimized':
        model = ResNetOptimized(args)
        print("使用优化版ResNet模型")
    elif model_type == 'baseline':
        model = ResNetBaseline(args)
        print("使用基线版ResNet模型")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 创建损失函数
    if args.loss == 'bce':
        print("使用标准BCE损失")
        criterion_cls = nn.BCEWithLogitsLoss()
    elif args.loss == 'wbce':
        print("使用加权BCE损失")
        criterion_cls = get_loss(
            type='wbce',
            class_instance_nums=args.class_instance_nums,
            total_instance_num=args.total_instance_num
        )
    elif args.loss == 'asl':
        print("使用ASL损失")
        # 如果配置中有ASL参数，使用它们
        gamma_neg = getattr(args, 'asl_gamma_neg', 4)
        gamma_pos = getattr(args, 'asl_gamma_pos', 1)
        clip = getattr(args, 'asl_clip', 0.05)
        
        criterion_cls = get_loss(
            type='asl',
            class_instance_nums=args.class_instance_nums,
            total_instance_num=args.total_instance_num
        )
    else:
        raise ValueError(f"Unsupported loss function: {args.loss}")

    model = model.to(device)   
    
    # 开始训练
    trainer = Trainer(model, criterion_cls, args, train_dataloader, val_dataloader, test_dataloader, device)
    trainer.train()

if __name__ == '__main__':
    # 可以通过修改这些参数来测试不同配置
    main(
        config='configs/resnet_optimized.yaml',
        stage='full',  # 先用dev测试，确认没问题后改为full
        model_type='optimized'  # 或者 'baseline' 进行对比
    ) 