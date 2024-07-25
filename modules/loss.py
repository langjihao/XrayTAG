import torch
import torch.nn as nn
'''
来自Chexfusion论文https://readpaper.com/pdf-annotate/note?pdfId=4786882020961681409&noteId=2353232875266081024

'''

def get_loss(type, class_instance_nums, total_instance_num):
    """
    根据指定的损失类型返回相应的损失函数对象。

    参数:
    - type (str): 损失函数的类型。可选值为 'bce'、'wbce' 或 'asl'。
        - 'bce': 二元交叉熵损失(Binary Cross-Entropy Loss)。
        - 'wbce': 带权重的二元交叉熵损失(Weighted Binary Cross-Entropy Loss)。
        - 'asl': 带权重的Asymmetric Loss(Asymmetric Loss with Class Weight)。
    - class_instance_nums (list of int): 每个类别的实例数量列表。
    - total_instance_num (int): 所有类别的总实例数量。

    返回:
    - nn.Module: 根据指定类型返回的PyTorch损失函数对象。

    异常:
    - ValueError: 如果指定的损失类型未知，则抛出此异常。

    示例:
    >>> get_loss('bce', [100, 200], 300)
    BCEWithLogitsLoss()

    >>> get_loss('wbce', [100, 200], 300)
    BCEwithClassWeights([100, 200], 300)

    >>> get_loss('asl', [100, 200], 300)
    ASLwithClassWeight([100, 200], 300)
    """
    if type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif type == 'wbce':
        return BCEwithClassWeights(class_instance_nums, total_instance_num)
    elif type == 'asl':
        return ASLwithClassWeight(class_instance_nums, total_instance_num)
    else:
        raise ValueError(f'Unknown loss type: {type}')


class BCEwithClassWeights(nn.Module):
    def __init__(self, class_instance_nums, total_instance_num):
        super(BCEwithClassWeights, self).__init__()
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32)
        p = class_instance_nums / total_instance_num
        self.pos_weights = torch.exp(1-p)
        self.neg_weights = torch.exp(p)

    def forward(self, pred, label):
        # https://www.cse.sc.edu/~songwang/document/cvpr21d.pdf (equation 4)
        weight = label * self.pos_weights.cuda() + (1 - label) * self.neg_weights.cuda()
        loss = nn.functional.binary_cross_entropy_with_logits(pred, label, weight=weight)
        return loss


class ASLwithClassWeight(nn.Module):
    def __init__(self, class_instance_nums, total_instance_num, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(ASLwithClassWeight, self).__init__()
        class_instance_nums = torch.tensor(class_instance_nums, dtype=torch.float32)
        p = class_instance_nums / total_instance_num
        self.pos_weights = torch.exp(1-p)
        self.neg_weights = torch.exp(p)
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, pred, label):
        weight = label * self.pos_weights.cuda() + (1 - label) * self.neg_weights.cuda()

        # Calculating Probabilities
        xs_pos = torch.sigmoid(pred)
        xs_neg = 1.0 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg.add_(self.clip).clamp_(max=1)

       # Basic CE calculation
        los_pos = label * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - label) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        loss *= weight

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * label 
            pt1 = xs_neg * (1 - label)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * label + self.gamma_neg * (1 - label)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()