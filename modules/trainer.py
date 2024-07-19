import os
from abc import abstractmethod
import torch
import pandas as pd
import numpy as np
from numpy import inf
import copy
from .optims import LinearWarmupCosineLRScheduler
from tqdm import tqdm
import swanlab

def compute_metrics(gts, res):
    gts_chexbert = np.array(gts)
    res_chexbert = np.array(res)
    res_chexbert = (res_chexbert == 1)
    gts_chexbert = (gts_chexbert == 1)

    tp = (res_chexbert * gts_chexbert).astype(float)

    fp = (res_chexbert * ~gts_chexbert).astype(float)
    fn = (~res_chexbert * gts_chexbert).astype(float)

    tp_eg = tp.sum(1)
    fp_eg = fp.sum(1)
    fn_eg = fn.sum(1)


    scores = {
        # example-based CE metrics
        'ce_precision': np.round(np.nan_to_num(tp_eg / (tp_eg + fp_eg)).mean(), 3),
        'ce_recall': np.round(np.nan_to_num(tp_eg / (tp_eg + fn_eg)).mean(), 3),
        'ce_f1': np.round(np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).mean(), 3),
        'ce_num_examples': float(len(res_chexbert)),
    }
    return scores

class BaseTrainer(object):
    def __init__(self, model, criterion_cls, base_probs, args, device):
        if args.stage != "dev":
            swanlab.init(
            # 设置将记录此次运行的项目信息
            project="MRG",
            # 跟踪超参数和运行元数据
            config=args,
            )

        self.args = args
        self.model = model
        self.device = device
        self.criterion_cls = criterion_cls
        self.base_probs = base_probs
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
            lr=float(self.args.init_lr),
            weight_decay=float(self.args.weight_decay),
            betas=(0.9, beta2),
        )
        #################

        self.epochs = self.args.epochs

        self.mnt_metric = 'val_' + args.monitor_metric

        self.mnt_best = 0 
        self.log_best = {}

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):

            result = self._train_epoch_blip(epoch)
            result = self.eval_blip(result)

            # save logged information 
            log = {'epoch': epoch}
            log.update(result)
            if self.args.stage != "dev":
                swanlab.log(log)
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
    def __init__(self, model, criterion_cls, base_probs,  args, train_dataloader, val_dataloader, test_dataloader, device):
        super(Trainer, self).__init__(model, criterion_cls, base_probs, args, device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.lr_scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer, 
            self.args.epochs, 
            self.args.min_lr, 
            self.args.init_lr, 
            decay_rate=None, 
            warmup_start_lr=self.args.warmup_lr,
            warmup_steps=self.args.warmup_steps,
        )

    def _train_epoch_blip(self, epoch):
        train_loss = 0
        self.model.train()
        for batch_idx, (images, cls_labels, clip_memory) in tqdm(enumerate(self.train_dataloader),total = len(self.train_dataloader)):
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            clip_memory = clip_memory.to(self.device)
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            preds = self.model(images, clip_memory,self.base_probs)
            loss = self.criterion_cls(preds, cls_labels)
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        return log

    def eval_blip(self, log):
        self.model.eval()
        logits = []
        counts = []
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images,cls_labels, clip_memory) in tqdm(enumerate(self.val_dataloader),total = len(self.val_dataloader)):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                clip_memory = clip_memory.to(self.device)
                preds,preds_logits = self.model.generate(images, clip_memory)
                
                val_gts += cls_labels.tolist()
                val_res += preds
                ## logit adjustment
                cls_labels = (cls_labels==1).float()
                logit = preds_logits*cls_labels
                logits.append(logit.cpu().numpy())
                counts.append(cls_labels.cpu().numpy())
            val_score = compute_metrics(val_gts, val_res)
            log.update(**{'val_' + k: v for k, v in val_score.items()})
            #######
            logits = np.concatenate(logits, axis=0)
            counts = np.concatenate(counts, axis=0)
            logits = np.sum(logits, 0)
            counts = np.sum(counts, 0)
            logits = logits / counts
            logits /= np.max(logits)
            logits = np.append(logits, [1,1,1,1]) # 4 auxiliary diseases
            #######
            self.base_probs = logits # update class distribution
            

        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images, cls_labels, clip_memory) in tqdm(enumerate(self.test_dataloader),total = len(self.test_dataloader)):
                images = images.to(self.device) 
                clip_memory = clip_memory.to(self.device) 
                preds,preds_logits = self.model.generate(images, clip_memory)
                test_gts += cls_labels.tolist()
                test_res += preds
            test_score = compute_metrics(test_gts, test_res)
            log.update(**{'test_' + k: v for k, v in test_score.items()})
        return log

    
if __name__ == '__main__':
    import random
    import math
    res = [ [3, 3, 3, 3, 2, 0, 0, 0, 1, 1, 2, 2, 2, 1, 3, 2, 2, 2],
            [2, 2, 2, 2, 1, 1, 0, 1, 3, 1, 1, 0, 1, 3, 0, 1, 2, 3],
            [0, 2, 3, 2, 0, 0, 0, 2, 3, 3, 2, 3, 3, 0, 2, 0, 2, 1],
            [2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 2, 1, 0, 3, 1, 2, 0, 1],
            [0, 1, 3, 1, 3, 3, 2, 0, 2, 0, 2, 0, 3, 2, 3, 3, 2, 1],
            [2, 2, 3, 3, 0, 3, 3, 2, 2, 0, 2, 2, 2, 0, 1, 2, 2, 1],
            [2, 0, 0, 0, 3, 0, 2, 2, 3, 2, 3, 1, 3, 3, 0, 2, 2, 3],
            [2, 0, 2, 2, 3, 2, 0, 1, 0, 0, 1, 1, 3, 2, 1, 3, 1, 2]]
    gts = [ [3, 3, 3, 3, 2, 0, 0, 0, 1, 1, 2, 2, 2, 1, 3, 2, 2, 2],
            [2, 2, 2, 2, 1, 1, 0, 1, 3, 1, 1, 0, 1, 3, 0, 1, 2, 3],
            [0, 2, 3, 2, 0, 0, 0, 2, 3, 3, 2, 3, 3, 0, 2, 0, 2, 1],
            [2, 0, 0, 2, 1, 2, 1, 1, 1, 2, 2, 1, 0, 3, 1, 2, 0, 1],
            [0, 1, 3, 1, 3, 3, 2, 0, 2, 0, 2, 0, 3, 2, 3, 3, 2, 1],
            [2, 2, 3, 3, 0, 3, 3, 2, 2, 0, 2, 2, 2, 0, 1, 2, 2, 1],
            [2, 0, 0, 0, 3, 0, 2, 2, 3, 2, 3, 1, 3, 3, 0, 2, 2, 3],
            [2, 0, 2, 2, 3, 2, 0, 1, 0, 0, 1, 1, 3, 2, 1, 3, 1, 2]]
    # 计算总元素数量
    total_elements = sum(len(row) for row in gts)

    # 确定要改乱的元素数量，这里以10%为例
    num_to_shuffle = math.floor(total_elements * 0.5)

    # 随机选择要改乱的元素的索引
    indices_to_shuffle = random.sample(range(total_elements), num_to_shuffle)

    # 对于每个选中的索引，随机生成一个新值并替换
    for index in indices_to_shuffle:
        # 计算行和列
        row = index // len(gts[0])
        col = index % len(gts[0])
        # 生成一个新值并替换，假设新值范围是0到3
        gts[row][col] = random.randint(0, 3)
    scores = compute_metrics(gts, res)
    print(scores)
    