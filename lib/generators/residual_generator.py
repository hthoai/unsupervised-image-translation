import torch.nn as nn
from torch import Tensor

from lib.layers import ConvNormRelu, ResidualBlock


class ResidualGenerator(nn.Module):
    """Resnet-based generator that consists of Residual blocks
    between a few downsampling/upsampling operations.
    
    Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix"""

    def __init__(
        self,
        nz: int,
        nc: int,
        ngf: int = 64,
        norm_type: str="batch",
        ng_blocks: int = 6,
        n_downsampling: int = 2
    ):
        """Construct a Resnet-based Generator.

        Parameters:
        -----------
            nz:             size of z latent vector
            nc:             the number of channels in output images
            ngf:            size of feature maps in generator
            norm_type:      normalization layer `batch` | `instance`
            n_blocks:       the number of Residual blocks
            n_downsampling: the number of encoder/decoder blocks
        """
        assert (ng_blocks >= 0)
        super(ResidualGenerator, self).__init__()
        # No need to use bias as BatchNorm2d has affine parameters
        bias = norm_type == "batch"
        ############################
        # ENCODER
        ############################
        encoder = [nn.ReflectionPad2d(3)]
        encoder += [
            ConvNormRelu(
                nz,
                ngf,
                conv_type="forward",
                norm_type=norm_type,
                kernel_size=7,
                bias=bias,
            )
        ]
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder += [
                ConvNormRelu(
                    ngf * mult,
                    ngf * mult * 2,
                    conv_type="forward",
                    norm_type=norm_type,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=bias,
                )
            ]
        self.encoder = nn.Sequential(*encoder)
        ############################
        # TRANSFORMATION
        ############################
        mult = 2 ** n_downsampling
        resblocks = []
        for i in range(ng_blocks):
            resblocks += [ResidualBlock(ngf * mult, norm_type, bias)]
        self.transform = nn.Sequential(*resblocks)
        ############################
        # DECODER
        ############################
        decoder = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [
                ConvNormRelu(
                    ngf * mult,
                    int(ngf * mult / 2),
                    conv_type="transpose",
                    norm_type=norm_type,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=bias,
                )
            ]
        decoder += [
            nn.Conv2d(
                ngf,
                nc,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            )
        ]
        decoder += [nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.transform(x)
        x = self.decoder(x)

        return x
    