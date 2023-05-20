import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Реализация ContrastiveLoss.

    """

    def __init__(self, margin=1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        pairwise_distances = torch.cdist(inputs, inputs, p=2)

        labels_equal = targets.unsqueeze(0) == targets.unsqueeze(1)
        labels_not_equal = ~labels_equal

        positive_mask = labels_equal.triu(diagonal=1)
        negative_mask = labels_not_equal.triu(diagonal=1)

        positive_distances = pairwise_distances.masked_select(positive_mask)

        if positive_distances.numel() > 0:
            positive_loss = positive_distances.pow(2).mean()
        else:
            positive_loss = 0.0

        negative_distances = pairwise_distances.masked_select(negative_mask)

        if negative_distances.numel() > 0:
            negative_loss = F.relu(self.margin - negative_distances).pow(2).mean()
        else:
            negative_loss = 0.0

        return positive_loss + negative_loss
