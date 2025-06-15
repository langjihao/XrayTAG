import torch
import torch.nn as nn
import torch.nn.functional as F
from models.VisionExtractor import VisionExtractor

class EnhancedClassificationHead(nn.Module):
    """增强的多标签分类头"""
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super(EnhancedClassificationHead, self).__init__()
        
        # 多层感知机分类头
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.BatchNorm1d(input_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            
            nn.Linear(input_dim // 4, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.classifier(x)

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [B, C, H, W]
        attention = self.conv(x)  # [B, 1, H, W]
        attention = self.sigmoid(attention)
        return x * attention

class ResNetOptimized(nn.Module):
    """优化的ResNet多标签分类模型"""
    def __init__(self, args):
        super(ResNetOptimized, self).__init__()
        
        # 视觉特征提取器
        self.vision_extractor = VisionExtractor(args)
        
        # 空间注意力机制（可选）
        if getattr(args, 'use_spatial_attention', False):
            self.spatial_attention = SpatialAttention(args.vision_channels)
        else:
            self.spatial_attention = None
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 增强的分类头
        if getattr(args, 'use_enhanced_head', True):
            dropout_rate = getattr(args, 'dropout_rate', 0.3)
            self.head = EnhancedClassificationHead(
                args.vision_channels, 14, dropout_rate
            )
        else:
            # 简单线性层（原版）
            self.head = nn.Linear(args.vision_channels, 14)
        
        # Dropout层
        dropout_rate = getattr(args, 'dropout_rate', 0.3)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # 特征提取
        patch_feats, avg_feats, patch = self.vision_extractor(x)
        
        # 可选：应用空间注意力
        if self.spatial_attention is not None:
            patch = self.spatial_attention(patch)
            # 重新计算全局特征
            avg_feats = self.global_pool(patch).flatten(1)
        
        # 应用dropout
        avg_feats = self.dropout(avg_feats)
        
        # 分类预测
        logits = self.head(avg_feats)
        cls_preds = torch.sigmoid(logits)
        
        return cls_preds, logits

class ResNetBaseline(nn.Module):
    """标准ResNet基线模型（用于对比）"""
    def __init__(self, args):
        super(ResNetBaseline, self).__init__()
        self.vision_extractor = VisionExtractor(args)
        self.head = nn.Linear(args.vision_channels, 14)
        
        # 降低dropout以提高基线性能
        dropout_rate = getattr(args, 'dropout_rate', 0.2)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        _, avg_feats, _ = self.vision_extractor(x)
        avg_feats = self.dropout(avg_feats)
        logits = self.head(avg_feats)
        cls_preds = torch.sigmoid(logits)
        return cls_preds, logits

def create_resnet_model(args, model_type='optimized'):
    """创建ResNet模型的工厂函数"""
    if model_type == 'optimized':
        return ResNetOptimized(args)
    elif model_type == 'baseline':
        return ResNetBaseline(args)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

if __name__ == '__main__':
    from types import SimpleNamespace
    
    # 测试模型
    args = SimpleNamespace(
        vision_model='resnet101',
        vision_channels=2048,
        image_size=224,
        use_enhanced_head=True,
        use_spatial_attention=True,
        dropout_rate=0.3
    )
    
    model = ResNetOptimized(args)
    x = torch.randn(2, 3, 224, 224)
    cls_preds, logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {cls_preds.shape}")
    print(f"Logits shape: {logits.shape}")
    print("Model created successfully!") 