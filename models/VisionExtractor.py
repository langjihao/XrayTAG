import torch
import torch.nn as nn
import timm

class VisionExtractor(nn.Module):
    '''
    return:
    patch_feats: NxLxD Bx49x2048
    avg_feats: NxL Bx2048
    patch: NxDxWxH Bx2048x7x7
    '''
    def __init__(self, args):
        super(VisionExtractor, self).__init__()
        model = timm.create_model(args.vision_model, pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        map_size = int(args.image_size / 32)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=map_size, stride=1, padding=0)
    
    def forward(self, x):
        patch = self.model(x)
        # NxL Bx2048
        avg_feats = self.avg_fnt(patch).flatten(1)
        batch_size, feat_size, _, _ = patch.shape
        # NxLxD Bx49x2048
        patch_feats = patch.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats, patch

