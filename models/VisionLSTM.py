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
        output_shape=(14,),
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
    def forward(self, x):
        preds = self.vision_lstm(x)
        cls_preds= torch.sigmoid(preds)
        return cls_preds, preds

