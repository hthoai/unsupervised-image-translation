import torch.nn as nn
from torch import Tensor


class ConvNormRelu(nn.Module):
    """Do convolution, normalize, (leaky) relu activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str,
        norm_layer: nn,
        kernel_size: int = 3,
        stride: int = 1,
        pad_size: int = 0,
        use_bias: bool = True,
        leaky: float = 0,
    ) -> None:
        super(ConvNormRelu, self).__init__()
        # Conv
        if conv_type == "forward":
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad_size,
                bias=use_bias,
            )
        elif conv_type == "transpose":
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad_size,
                output_padding=pad_size,
                bias=use_bias,
            )
        else:
            raise NotImplementedError(f"conv_type {conv_type} is not implemented.")
        # Normalize
        self.norm = norm_layer(out_channels)
        # (Leaky) Relu
        if leaky > 0:
            self.relu = nn.LeakyReLU(negative_slope=leaky)
        else:
            self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x
