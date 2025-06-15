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
    # 适用于大部分网络
    def __init__(self, args):
        super(VisionExtractor, self).__init__()
        model = timm.create_model(args.vision_model, pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        map_size = int(args.image_size / 32)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=map_size, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.5) 
    def forward(self, x):
        patch = self.model(x)
        patch = self.dropout(patch)
        # NxL Bx2048
        avg_feats = self.avg_fnt(patch).flatten(1)
        batch_size, feat_size, _, _ = patch.shape
        # NxLxD Bx49x2048
        patch_feats = patch.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        
        return patch_feats, avg_feats, patch

    # 适用于swin_tiny_patch4_window7_224
#     def __init__(self, args):
#         super(VisionExtractor, self).__init__()
#         self.model = timm.create_model(args.vision_model, pretrained=True)
        
#         # 移除最后的分类头
#         self.model.head = nn.Identity()
        
#         # 获取嵌入维度
#         self.embed_dim = self.model.embed_dim
        
#     def forward(self, x):
#         # x的输入形状为 [B, 3, 224, 224]
        
#         # 使用整个模型进行前向传播
#         x = self.model.forward_features(x)  # [B, 49, 768]
        
#         # patch_feats: NxLxD (Bx7x7x768)
#         patch_feats = x
        
#         # avg_feats: NxD (Bx768)
#         avg_feats = x.mean(dim=[1, 2])
        
#         # patch: NxDxHxW (Bx768x7x7)
#         patch = x.permute(0, 3, 1, 2)
#         return patch_feats, avg_feats, patch

# if __name__ == '__main__':
#     args = argparse.Namespace(vision_model='swin_tiny_patch4_window7_224')
#     vision_extractor = VisionExtractor(args)
#     x = torch.randn(1, 3, 224, 224)
#     patch_feats, avg_feats, patch = vision_extractor(x)
#     print(patch_feats.shape, avg_feats.shape, patch.shape)
