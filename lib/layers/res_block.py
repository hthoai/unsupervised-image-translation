import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(
        self,
        dim: int,
        padding_mode: str,
        norm_layer: nn,
        use_bias: bool = True,
        leaky: float = 0,
    ):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResBlock, self).__init__()
        if padding_mode == "reflect":
            self.conv = nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                padding_mode="reflect",
                bias=use_bias,
            )
        elif padding_mode == "replicate":
            self.conv = nn.Conv2d(
                dim,
                dim,
                kernel_size=3,
                padding=1,
                padding_mode="replicate",
                bias=use_bias,
            )
        else:  # padding_mode == 'zeros'
            self.conv = nn.Conv2d(
                dim, dim, kernel_size=3, padding=1, padding_mode="zeros", bias=use_bias
            )
        self.norm = norm_layer(dim)
        if leaky > 0:
            self.relu = nn.LeakyReLU(negative_slope=leaky)
        else:
            self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function (with skip connections)"""
        out = self.conv(x)
        out = self.norm(x)
        out = self.relu(out)

        out = self.conv(out)
        out = self.norm(x)
        out = x + out

        return out
