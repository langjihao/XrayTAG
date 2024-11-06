import torch
import torch.nn as nn
from models.VisionExtractor import VisionExtractor
from models.ml_decoder import MLDecoder
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer
import torch.nn.functional as F

class Chexfusion(nn.Module):
    def __init__(self, args):
        super(Chexfusion, self).__init__()
        self.vision_extractor = VisionExtractor(args)
        self.pos_encoding = Summer(PositionalEncoding2D(args.vision_channels))
        self.clshead = MLDecoder(num_classes=28, initial_num_features=args.vision_channels)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def forward(self, x, probs):
        _ , _ , patch = self.vision_extractor(x)
        patch_feats = self.pos_encoding(patch)
        preds = self.clshead(patch_feats)
        preds = preds.view(-1, 2, 14)
        cls_preds= F.softmax(preds, dim=1)
        cls_preds = cls_preds[:, 1, :]
        # logit adjustment
        preds[:, 1, :] += torch.log(torch.from_numpy(probs)).view(1, -1).to(self.device)
        #cls_preds是结果,preds是logits用于计算loss
        return cls_preds, preds
    def generate(self, x):
        _ , _ , patch = self.vision_extractor(x)
        patch_feats = self.pos_encoding(patch)
        preds = self.clshead(patch_feats)
        preds = preds.view(-1, 2, 14)
        cls_preds = F.softmax(preds, dim=1)
        logits = cls_preds[:, 1, :]
        #cls_preds是结果,preds用于算子调整
        return logits, logits

if __name__ == '__main__':
    from box import Box
    args = Box({
        'vision_model': 'convnext_small.in12k_ft_in1k',
        'vision_chanels': 768,
        'image_size': 224
    })

    model = Chexfusion(args)
    x = torch.randn(1, 3, 224, 224)
    y= model(x)
    print(y)