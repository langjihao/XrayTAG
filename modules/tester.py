import os
from abc import abstractmethod
import torch
import pandas as pd
import numpy as np
from numpy import inf
import copy
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

class BaseTester(object):
    def __init__(self, model,args, device):
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
        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir
    @abstractmethod
    def test(self):
        raise NotImplementedError
    @abstractmethod
    def plot(self):
        raise NotImplementedError
class Tester(BaseTester):
    def __init__(self, model, args, test_dataloader, device):
        super(Tester, self).__init__(model, args, device)

        self.test_dataloader = test_dataloader

    def test_blip(self):
        self.model.eval()
        log = {}
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
        print(log)
        return log

    