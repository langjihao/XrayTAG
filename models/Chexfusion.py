import torch
import torch.nn as nn
from models.VisionExtractor import VisionExtractor
from models.ml_decoder import MLDecoder
from positional_encodings.torch_encodings import PositionalEncoding2D, Summer

class Chexfusion(nn.Module):
    def __init__(self, args):
        super(Chexfusion, self).__init__()
        self.vision_extractor = VisionExtractor(args)
        self.pos_encoding = Summer(PositionalEncoding2D(args.vision_chanels))
        self.clshead = MLDecoder(num_classes=14, initial_num_features=args.vision_chanels)
    def forward(self, x):
        _ , _ , patch = self.vision_extractor(x)
        patch_feats = self.pos_encoding(patch)
        preds = self.clshead(patch_feats)
        return preds

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