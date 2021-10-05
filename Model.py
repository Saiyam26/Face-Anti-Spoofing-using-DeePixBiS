import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
# import torch.sigmoid as sigmoid


class DeePixBiS(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        dense = models.densenet161(pretrained=pretrained)
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[:8])
        self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(14 * 14, 1)

    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        out_map = F.sigmoid(dec)
        # print(out_map.shape)
        out = self.linear(out_map.view(-1, 14 * 14))
        out = F.sigmoid(out)
        out = torch.flatten(out)
        return out_map, out
