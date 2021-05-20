import torch
import torch.nn as nn


class GaussianNoiseLayer(nn.Module):

    def __init__(self,):
        super(GaussianNoiseLayer, self).__init__()

    def forward(self, x):
        if self.training == False:
            return x
        noise = torch.randn(x.size())

        return x + noise
