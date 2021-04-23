import torch.nn as nn
from torch import Tensor

from lib.layers import ConvNormRelu, ResidualBlock


class ResidualGenerator(nn.Module):
    """Resnet-based generator that consists of Residual blocks
    between a few downsampling/upsampling operations
    adapted Torch code and idea from Justin Johnson's neural style transfer project
    https://github.com/jcjohnson/fast-neural-style"""

    def __init__(
        self,
        nz: int,
        nc: int,
        ngf: int = 64,
        norm_layer: nn = nn.BatchNorm2d,
        ng_blocks: int = 6,
    ) -> None:
        """Construct a Resnet-based Generator.

        Parameters:
        -----------
            nz:          size of z latent vector
            nc:          the number of channels in output images
            ngf:         size of feature maps in generator
            norm_layer:  normalization layer
            n_blocks:    the number of Residual blocks
        """
        assert (ng_blocks >= 0)
        super(ResidualGenerator, self).__init__()
        # No need to use bias as BatchNorm2d has affine parameters
        use_bias = norm_layer == nn.InstanceNorm2d
        # ENCODING
        encoder = [nn.ReflectionPad2d(3)]
        encoder += [
            ConvNormRelu(
                nz,
                ngf,
                conv_type="forward",
                norm_layer=norm_layer,
                kernel_size=7,
                use_bias=use_bias,
            )
        ]
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder += [
                ConvNormRelu(
                    ngf * mult,
                    ngf * mult * 2,
                    conv_type="forward",
                    norm_layer=norm_layer,
                    kernel_size=3,
                    stride=2,
                    pad_size=1,
                    use_bias=use_bias,
                )
            ]
        self.encoder = nn.Sequential(*encoder)
        # TRANSFORMATION
        mult = 2 ** n_downsampling
        resblocks = []
        for i in range(ng_blocks):
            resblocks += [ResidualBlock(ngf * mult, norm_layer, use_bias)]
        self.transform = nn.Sequential(*resblocks)
        # DECODING
        decoder = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [
                ConvNormRelu(
                    ngf * mult,
                    int(ngf * mult / 2),
                    conv_type="transpose",
                    norm_layer=norm_layer,
                    kernel_size=3,
                    stride=2,
                    pad_size=1,
                    use_bias=use_bias,
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
