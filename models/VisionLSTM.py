from models.vision_lstm import VisionLSTM2
import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
class VisionLSTM(nn.Module):
    def __init__(self,args):
        super(VisionLSTM, self).__init__()
        self.vision_lstm = VisionLSTM2(
        dim=384,
        depth=12,
        legacy_norm=True,
        output_shape=(28,),
        mode="classifier",
        pooling="bilateral_flatten",
        conv_kind="2d",
        conv_kernel_size=3,
        norm_bias=True,
        proj_bias=True,
        )
        state_dict = torch.load(args.load_pretrained, map_location="cpu")

        # 获取当前模型的状态字典
        model_dict = self.vision_lstm.state_dict()

        # 过滤掉不匹配的参数
        filtered_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].size() == v.size():
                filtered_state_dict[k] = v

        # 更新当前模型的状态字典
        model_dict.update(filtered_state_dict)

        # 加载更新后的状态字典
        self.vision_lstm.load_state_dict(model_dict, strict=False)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def forward(self, x, probs):
        preds = self.vision_lstm(x)
        preds = preds.view(-1, 2, 14)
        cls_preds= F.softmax(preds, dim=1)
        cls_preds = cls_preds[:, 1, :]
        # logit adjustment
        preds[:, 1, :] += torch.log(torch.from_numpy(probs)).view(1, -1).to(self.device)
        #cls_preds是结果,preds是logits用于计算loss
        return cls_preds, preds
    def generate(self, x):
        preds = self.vision_lstm(x)
        preds = preds.view(-1, 2, 14)
        cls_preds = F.softmax(preds, dim=1)
        logits = cls_preds[:, 1, :]
        #cls_preds是结果,preds用于算子调整
        return logits, logits
if __name__ == '__main__':
    import numpy as np
    from box import Box
    args = Box({
        'load_pretrained': 'checkpoints/vision_lstm.pth',
    })
    model = VisionLSTM(args)
    x = torch.randn(1, 3, 224, 224)
    base_probs = [
    0.05507588906532738,
    0.2219210458288711,
    0.24894198456368405,
    0.03815133498282802,
    0.07480335315188892,
    0.02870859337493999,
    0.03160013294434802,
    0.2093688836367665,
    0.02353853539643266,
    0.18073045533439197,
    0.0206359171313564,
    0.037213338749584546,
    0.22214261974223568,
    0.23941430628900623
    ]
    probs = np.array(base_probs) / np.max(base_probs)