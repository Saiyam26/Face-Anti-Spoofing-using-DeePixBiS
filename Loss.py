import torch
from torch import nn
import torch.nn.functional as F


class PixWiseBCELoss(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.beta = beta

    def forward(self, net_mask, net_label, target_mask, target_label):
        pixel_loss = self.criterion(net_mask, target_mask)
        binary_loss = self.criterion(net_label, target_label)
        loss = pixel_loss * self.beta + binary_loss * (1 - self.beta)
        return loss
