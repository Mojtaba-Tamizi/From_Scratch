import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()