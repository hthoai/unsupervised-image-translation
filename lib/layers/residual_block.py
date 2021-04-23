import torch.nn as nn
from torch import Tensor


class ResidualBlock(nn.Module):
    """Define a Residual block"""

    def __init__(
        self,
        dim: int,
        norm_layer: nn,
        use_bias: bool = True,
        leaky: float = 0,
    ) -> None:
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
            bias=use_bias,
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

        return x + out
