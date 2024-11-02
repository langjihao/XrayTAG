from models.vision_lstm import VisionLSTM2
import torch.nn as nn
import torch

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
    def forward(self, x):
        preds = self.vision_lstm(x)
        return preds