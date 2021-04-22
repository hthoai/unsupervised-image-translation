import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ConvNormRelu(nn.Module):
    """Do convolution, normalize, (leaky) relu activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_type: str,
        norm_layer: nn,
        kernel_size: int=3,
        stride: int=1,
        pad_size: int=0,
        padding_mode: str='zeros',
        use_bias: bool=False,
        leaky: bool=False,
    ):
        super(ConvNormRelu, self).__init__()
        if padding_mode == "reflect":
            if conv_type == "upsampling":
                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad_size,
                    padding_mode="reflect",
                    bias=use_bias,
                )
            else: # conv_type == "downsampling"
                self.conv = nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad_size,
                    padding_mode="reflect",
                    bias=use_bias,
                )
        elif padding_mode == "replicate":
            if conv_type == "upsampling":
                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad_size,
                    padding_mode="replicate",
                    bias=use_bias,
                )
            else: # conv_type == "downsampling"
                self.conv = nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad_size,
                    padding_mode="replicate",
                    bias=use_bias,
                )
        else:  # padding_type == 'zeros'
            if conv_type == "upsampling":
                self.conv = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad_size,
                    padding_mode="zeros",
                    bias=use_bias,
                )
            else: # conv_type == "downsampling"
                self.conv = nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad_size,
                    padding_mode="zeros",
                    bias=use_bias,
                )
        self.norm = norm_layer(out_channels)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x
