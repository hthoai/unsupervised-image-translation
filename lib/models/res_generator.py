import torch.nn as nn
from torch import Tensor

from lib.layers import ConvNormRelu, ResBlock


class ResGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks
    between a few downsampling/upsampling operations
    adapted Torch code and idea from Justin Johnson's neural style transfer project
    https://github.com/jcjohnson/fast-neural-style"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ng_filters: int = 64,
        norm_layer: nn = nn.BatchNorm2d,
        n_blocks: int = 6,
        padding_mode: str = "reflect",
    ):
        """Construct a Resnet-based Generator.

        Parameters:
        -----------
            in_channels:    the number of channels in input images
            out_channels:   the number of channels in output images
            ng_filters:     the number of filters in the last conv layer
            norm_layer:     normalization layer
            n_blocks:       the number of ResNet blocks
            padding_type:   the name of padding layer in conv layers
                            `reflect` | `replicate` | `zeros`
        """
        assert n_blocks >= 0
        super(ResGenerator, self).__init__()
        # No need to use bias as BatchNorm2d has affine parameters
        use_bias = norm_layer == nn.InstanceNorm2d
        self.conv_norm_relu = ConvNormRelu(
            in_channels,
            ng_filters,
            conv_type="forward",
            norm_layer=norm_layer,
            use_bias=use_bias,
        )
        # ENCODING
        n_downsampling = 2
        encoder = []
        for i in range(n_downsampling):
            mult = 2 ** i
            encoder += [
                ConvNormRelu(
                    ng_filters * mult,
                    ng_filters * mult * 2,
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
        for i in range(n_blocks):
            resblocks += [
                ResBlock(ng_filters * mult, padding_mode, norm_layer, use_bias)
            ]
        self.transform = nn.Sequential(*resblocks)
        # DECODING
        decoder = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            decoder += [
                ConvNormRelu(
                    ng_filters * mult,
                    int(ng_filters * mult / 2),
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
                ng_filters,
                out_channels,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            )
        ]
        decoder += [nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_norm_relu(x)
        x = self.encoder(x)
        x = self.transform(x)
        x = self.decoder(x)

        return x
