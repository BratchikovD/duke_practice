import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcFaceLoss(nn.Module):
    """Реализация ArcFace.

    Reference: ArcFace: Additive Angular Margin Loss for Deep Face Recognition arXiv:1801.07698
    """

    def __init__(self, in_features=1024, out_features=702, s=30, m=0.5, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels):
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return F.cross_entropy(output, labels)
