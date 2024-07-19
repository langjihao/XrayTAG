import numpy as np
from collections import OrderedDict
from transformers import BertConfig, BertModel, BertTokenizer, logging
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")
CONDITIONS = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]

class CheXbert(nn.Module):
    '''
    CheXbert纯模型版
    初始化无参数
    forward接受tokenized，返回14个类别的预测
    '''
    def __init__(self):
        super(CheXbert, self).__init__()
        config = BertConfig().from_pretrained('bert-base-uncased')
        self.bert = BertModel(config)
        # 768
        hidden_size = 768
        # Classes: present, absent, unknown, blank for 12 conditions + support devices
        self.linear_heads = nn.ModuleList([nn.Linear(hidden_size, 4, bias=True) for _ in range(13)])
        # Classes: yes, no for the 'no finding' observation
        self.linear_heads.append(nn.Linear(hidden_size, 2, bias=True))

    def forward(self, tokenized):
        last_hidden_state = self.bert(**tokenized)[0]
        cls = last_hidden_state[:, 0, :]
        # cls = self.dropout(cls)

        predictions = []
        for i in range(14):
            predictions.append(self.linear_heads[i](cls).argmax(dim=1))
        # Nx14
        return torch.stack(predictions, dim=1)

class CheXbertLabeler():
    ''' 
    CheXbert标签器
    加载CheXbert模型权重
    fordward接受文本，返回14个类别的预测
    '''
    def __init__(self, checkpoint_path, device, mode='compute'):
        super(CheXbertLabeler, self).__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.chexbert = CheXbert().to(self.device)
        self.chexbert.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        self.chexbert.eval()
        self.mode = mode

    def tensor_to_classified_list(self, tensor, conditions=CONDITIONS):
        classified_list = []
        for row in tensor:
            class_dict = {condition: 0 for condition in conditions}  # 初始化字典
            for i in range(len(row)) :
                class_dict[conditions[i]] = row[i].item()  # 更新计数
            classified_list.append(class_dict)
        return classified_list
    
    def labeler(self, reports):
        for i in range(len(reports)):
            reports[i] = reports[i].strip()
            reports[i] = reports[i].replace(r"\n", " ")
            reports[i] = reports[i].replace(r"\s+", " ")
            reports[i] = reports[i].replace(r"\s+(?=[\.,])", "")
            reports[i] = reports[i].strip()
        tokenized = self.tokenizer(reports, padding='longest', return_tensors="pt")
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        result = self.chexbert(tokenized)
        if self.mode == 'compute':
            return result
        else:
            return self.tensor_to_classified_list(result)
     
class CheXbertMetrics():
    '''
    @param gts: list of ground truth reports
    @param res: list of predicted reports
    @return: dict of metrics
    '''
    def __init__(self, checkpoint_path = '/root/OriginPromptMRG/PromptMRG_cls/checkpoints/checkpoint.pth', mbatch_size=16, device='cuda'):
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.device = device
        self.chexbert = CheXbertLabeler(self.checkpoint_path, self.device)

    def mini_batch(self, gts, res, mbatch_size=16):
        length = len(gts)
        assert length == len(res)
        for i in range(0, length, mbatch_size):
            yield gts[i:min(i + mbatch_size, length)], res[i:min(i + mbatch_size, length)]
    # TODO 为什么gts不能用label中的进行替换，反而要重新执行一遍计算过程呢
    def compute(self, gts, res):
        gts_chexbert = []
        res_chexbert = []
        for gt, re in self.mini_batch(gts, res, self.mbatch_size):
            gt_chexbert = self.chexbert.labeler(list(gt)).tolist()
            re_chexbert = self.chexbert.labeler(list(re)).tolist()
            gts_chexbert += gt_chexbert
            res_chexbert += re_chexbert
        gts_chexbert = np.array(gts_chexbert)
        res_chexbert = np.array(res_chexbert)
        print(res_chexbert)
        res_chexbert = (res_chexbert == 1)
        gts_chexbert = (gts_chexbert == 1)

        tp = (res_chexbert * gts_chexbert).astype(float)

        fp = (res_chexbert * ~gts_chexbert).astype(float)
        fn = (~res_chexbert * gts_chexbert).astype(float)

        tp_cls = tp.sum(0)
        fp_cls = fp.sum(0)
        fn_cls = fn.sum(0)

        tp_eg = tp.sum(1)
        fp_eg = fp.sum(1)
        fn_eg = fn.sum(1)

        # precision_class = np.nan_to_num(tp_cls / (tp_cls + fp_cls))
        # recall_class = np.nan_to_num(tp_cls / (tp_cls + fn_cls))
        # f1_class = np.nan_to_num(tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls)))

        scores = {
            # example-based CE metrics
            'ce_precision': np.nan_to_num(tp_eg / (tp_eg + fp_eg)).mean(),
            'ce_recall': np.nan_to_num(tp_eg / (tp_eg + fn_eg)).mean(),
            'ce_f1': np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).mean(),
            'ce_num_examples': float(len(res_chexbert)),
            # class-based CE metrics
            # 'ce_precision_class': precision_class,
            # 'ce_recall_class': recall_class,
            # 'ce_f1_class': f1_class,
        }
        return scores

if __name__ == '__main__':
    import pprint

    # checkpoint = 'checkpoint.pth'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # chexbert = CheXbertLabeler(checkpoint, device)
    checkpoint = '/root/OriginPromptMRG/PromptMRG_cls/checkpoints/checkpoint.pth'
    res =  [
        'Frontal and lateral radiographs of the chest.  There is no obvious lobar\n airspace consolidation.  Increased perihilar opacities and interstitial\n markings are consistent with mild pulmonary edema.  The heart size is\n minimally enlarged.  There is no pneumothorax or pleural effusion.  Although\n the patient is somewhat rotated, rightward deviation of the trachea is likely\n secondary to tortuous aorta.  Marked kyphosis of the spine is unchanged. \n There is a stable moderate-large hiatal hernia.',
        'The heart is normal in size. The mediastinum is unremarkable. The lungs are clear.',
    ]
    gth = [
        'Frontal and lateral radiographs of the chest.  There is no obvious lobar\n airspace consolidation.  Increased perihilar opacities and interstitial\n markings are consistent with mild pulmonary edema.  The heart size is\n minimally enlarged.  There is no pneumothorax or pleural effusion.  Although\n the patient is somewhat rotated, rightward deviation of the trachea is likely\n secondary to tortuous aorta.  Marked kyphosis of the spine is unchanged. \n There is a stable moderate-large hiatal hernia.',
        'The heart is normal in size. The mediastinum is unremarkable. The lungs are clear.',
    ]
    score = CheXbertMetrics(checkpoint_path=checkpoint).compute(gth, res)
    pprint.pprint(score)
    # checkpoint = '/root/OriginPromptMRG/PromptMRG_cls/checkpoints/checkpoint.pth'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # chexbert = CheXbertLabeler(checkpoint, device,mode='label')
    # label = chexbert.labeler(gth)
    # print(label)
    