from __future__ import division, absolute_import
import torch
import torch.nn as nn


class ArcfaceLoss(nn.Module):
    """Реализация ArcFace.

    Reference: ArcFace: Additive Angular Margin Loss for Deep Face Recognition arXiv:1801.07698
    """
    def __init__(self):
        super(ArcfaceLoss, self).__init__()

    def forward(self):
        pass