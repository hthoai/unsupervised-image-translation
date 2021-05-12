import torch.nn as nn
from torch import Tensor

from utils.get_norm_layer import get_norm_layer


class ResidualBlock(nn.Module):
    """Define a Residual block"""

    def __init__(
        self,
        dim: int,
        norm_type: str,
        bias: bool = True,
        leaky: float = 0,
    ):
        """Initialize the Residual Block

        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            bias=bias,
        )
        self.norm = get_norm_layer(norm_type)(dim)
        if leaky > 0:
            self.relu = nn.LeakyReLU(negative_slope=leaky, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function (with skip connections)"""

        out = self.conv(x)
        out = self.norm(x)
        out = self.relu(out)

        out = self.conv(out)
        out = self.norm(x)

        return x + out
