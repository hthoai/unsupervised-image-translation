import torch.nn as nn
from torch import Tensor

from utils.get_norm_layer import get_norm_layer


class ConvNormRelu(nn.Module):
    """Do conv(transpose), (normalize), (leaky) relu activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str,
        norm_type: None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        leaky: float = 0,
    ):
        super(ConvNormRelu, self).__init__()
        layers = []
        # Conv
        if conv_type == "forward":
            layers += [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            ]
        elif conv_type == "transpose":
            layers += [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=padding,
                    bias=bias,
                )
            ]
        else:
            raise NotImplementedError(f"conv_type {conv_type} is not implemented.")
        # (Normalize)
        if norm_type is not None:
            layers += [get_norm_layer(norm_type)(out_channels)]
        # (Leaky) Relu
        if leaky > 0:
            layers += [nn.LeakyReLU(negative_slope=leaky, inplace=True)]
        else:
            layers += [nn.ReLU(inplace=True)]
        self.apply_layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.apply_layers(x)

        return x
