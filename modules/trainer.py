import os
from abc import abstractmethod
import torch
import pandas as pd
import numpy as np
from numpy import inf
import copy
# from .optims import LinearWarmupCosineLRScheduler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import swanlab
from torchmetrics import AveragePrecision, AUROC
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelAUROC,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score
)
class Metrics:
    def __init__(self, num_classes, device='cpu'):
        self.device = device
        self.map_metric = MultilabelAveragePrecision(num_labels=num_classes, average='macro')
        self.auroc_metric = MultilabelAUROC(num_labels=num_classes, average='macro')
        self.precision_metric = MultilabelPrecision(num_labels=num_classes, average='macro')
        self.recall_metric = MultilabelRecall(num_labels=num_classes, average='macro')
        self.f1_metric = MultilabelF1Score(num_labels=num_classes, average='macro')
    def compute(self,labels,preds):
        if isinstance(labels, list):
            labels = torch.stack(labels)
        if isinstance(preds, list):
            preds = torch.stack(preds)
        labels = labels.to(self.device)
        preds = preds.to(self.device)
        map_score = self.map_metric(preds, labels)
        auroc_score = self.auroc_metric(preds, labels)
        precision_score = self.precision_metric(preds, labels)
        recall_score = self.recall_metric(preds, labels)
        f1_score = self.f1_metric(preds, labels)
        return {
            'mAP': map_score,
            'AUC': auroc_score,
            'Precision': precision_score,
            'Recall': recall_score,
            'F1': f1_score
        }


    
class BaseTrainer(object):
    def __init__(self, model, criterion_cls, args, device):
        self.args = args
        self.model = model
        self.device = device
        self.criterion_cls = criterion_cls
        #################
        self.optimizer = None
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        print("number of trainable parameters: {}".format(num_parameters))
        args.num_parameters = num_parameters
        if args.stage != "dev":
            swanlab.init(
            # 设置将记录此次运行的项目信息
            project=args.project,
            experiment_name=args.experiment_name,
            description = args.description,
            # 跟踪超参数和运行元数据
            config=args,
            )
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(self.args.weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        beta2 = 0.999
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.args.lr),
            weight_decay=float(self.args.weight_decay),
            betas=(0.9, beta2),
        )
        #################

        self.epochs = self.args.epochs

        self.mnt_metric = 'val_' + args.monitor_metric

        self.mnt_best = 0 
        self.log_best = {}

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir + self.args.experiment_name + '/'
        self.metric = Metrics(14)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):

            result = self._train_step(epoch)
            result = self.eval_step(result)

            # save logged information 
            log = {'epoch': epoch}
            log.update(result)
            if self.args.stage != "dev":
                swanlab.log(log)
            # save current model
            torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'current.pth'))
            # record best
            if log[self.mnt_metric] >= self.mnt_best:
                self.mnt_best = log[self.mnt_metric]
                self.log_best = copy.deepcopy(log)
                best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
                torch.save(self.model.state_dict(), best_path)
                print("Saving current best to {}".format(best_path))

            # print logged information 
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))
        print('Best results w.r.t {}:'.format(self.mnt_metric))
        for key, value in self.log_best.items():
            print('\t{:15s}: {}'.format(str(key), value))

class Trainer(BaseTrainer):
    def __init__(self, model, criterion_cls, args, train_dataloader, val_dataloader, test_dataloader, device):
        super(Trainer, self).__init__(model, criterion_cls, args, device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=self.args.epochs*len(self.train_dataloader))
    def _train_step(self, epoch):
        train_loss = 0
        self.model.train()
        for batch_idx, (images, cls_labels) in tqdm(enumerate(self.train_dataloader),total = len(self.train_dataloader)):
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            preds = self.model(images)
            # cls_labels.shape = (N, 14)
            loss = self.criterion_cls(preds, cls_labels)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        return log

    def eval_step(self, log):
        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images,cls_labels) in tqdm(enumerate(self.val_dataloader),total = len(self.val_dataloader)):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                preds = self.model(images)
                val_gts += cls_labels
                val_res += preds
            val_score = self.metric.compute(val_gts, val_res)
            log.update(**{'val_' + k: v for k, v in val_score.items()})
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images, cls_labels) in tqdm(enumerate(self.test_dataloader),total = len(self.test_dataloader)):
                images = images.to(self.device) 
                preds = self.model(images)
                test_gts += cls_labels
                test_res += preds
            test_score = self.metric.compute(test_gts, test_res)
            log.update(**{'test_' + k: v for k, v in test_score.items()})
        return log

    