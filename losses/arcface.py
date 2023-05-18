from __future__ import absolute_import, division

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcfaceLoss(nn.Module):
    """Реализация ArcFace.

    Reference: ArcFace: Additive Angular Margin Loss for Deep Face Recognition arXiv:1801.07698
    """
    def __init__(self, in_features, out_features, feature_scale=30.0, margin=0.50):
        super(ArcfaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feature_scale = feature_scale
        self.margin = margin
        self.cosine_threshold = math.cos(math.pi - margin)
        self.modified_margin = math.sin(math.pi - margin) * margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs, targets):
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        sin_theta = torch.sqrt((1.0-torch.pow(cosine, 2)).clamp(0, 1))

        phi = cosine*math.cos(self.margin) - sin_theta*math.sin(self.margin)
        phi = torch.where(cosine > self.cosine_threshold, phi, cosine - self.modified_margin)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.feature_scale

        cross_entropy = nn.CrossEntropyLoss()

        return cross_entropy(output, targets)
