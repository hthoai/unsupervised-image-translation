import torch.nn as nn
from torch import Tensor
from lib import layers
from lib.layers import ConvNormRelu


class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN (just a convnet:)) discriminator."""

    def __init__(
        self,
        nc: int,
        kernel_size: int=4,
        ndf: int = 64,
        nd_layers: int = 3,
        norm_type: str="batch",
    ) -> None:
        """Construct a PatchGAN discriminator.

        Parameters:
        -----------
            nc:             the number of channels in output images
            kernel_size:    convolution kernel size
            ndf:            size of feature maps in discriminator
            n_layers:       the number of conv layers in the discriminator
            norm_type:      normalization layer `batch` | `instance`
        """
        super(PatchDiscriminator, self).__init__()
        # No need to use bias as BatchNorm2d has affine parameters
        use_bias = norm_type == "instance"
        padw = 1
        self.conv1 = nn.Conv2d(nc, ndf, kernel_size=kernel_size, stride=2, padding=padw)
        self.leaky_relu = nn.LeakyReLU(0.2)
        nf_mult = 1
        nf_mult_prev = 1
        layers = []
        # Gradually increase the number of filters
        for n in range(1, nd_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [
                ConvNormRelu(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    conv_type="forward",
                    norm_type=norm_type,
                    kernel_size=kernel_size,
                    stride=2,
                    pad_size=padw,
                    use_bias=use_bias,
                    leaky=0.2,
                )
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** nd_layers, 8)
        self.conv_norm_relus = nn.Sequential(*layers)
        self.conv2 = ConvNormRelu(
            ndf * nf_mult_prev,
            ndf * nf_mult,
            conv_type="forward",
            norm_type=norm_type,
            kernel_size=kernel_size,
            stride=1,
            pad_size=padw,
            use_bias=use_bias,
            leaky=0.2,
        )
        self.output = nn.Conv2d(
            ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padw
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv_norm_relus(x)
        x = self.conv2(x)
        x = self.output(x)

        return x
