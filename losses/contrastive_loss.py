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
        n = inputs.size(0)
        # Compute similarity matrix
        sim_matrix = F.cosine_similarity(inputs.unsqueeze(1), inputs.unsqueeze(0), dim=2)
        # Create positive and negative mask
        target_matrix = targets.view(n, 1) == targets.view(1, n)
        positive_mask = target_matrix.triu(diagonal=1)  # Exclude self-comparison
        negative_mask = ~target_matrix
        # Compute positive and negative loss
        positive_loss = (1 - sim_matrix).pow(2).masked_select(positive_mask).mean()
        negative_loss = F.relu(sim_matrix - self.margin).pow(2).masked_select(negative_mask).mean()

        return positive_loss + negative_loss
