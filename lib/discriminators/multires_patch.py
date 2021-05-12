from typing import List, Tuple
from torch import nn
from torch.functional import Tensor

from .patch import PatchDiscriminator


class WeightSharedMultiResPatchDiscriminator(nn.Module):
    """Weight-shared multi-resolution patch discriminator with shared weights.

    Adapted from 
    """

    def __init__(self,
                 num_D: int=3,
                 kernel_size: int=3,
                 nc: int=3,
                 ndf: int=64,
                 nd_layers: int=4,
                 norm_type: str="batch") -> None:
        """Construct a WeightSharedMultiResPatchDiscriminator.

        Parameters:
        -----------
            `num_D`:          num. of discriminators (one per scale).
            `kernel_size`:    convolution kernel size.
            `nc`:             num. of channels in the real/fake image.
            `ndf`:            num. of base filters in a layer.
            `nd_layers`:      num. of layers for the patch discriminator.
            `norm_type`:      batch_norm/instance_norm/none/....
        """
        super().__init__()
        self.num_D = num_D
        self.discriminator = PatchDiscriminator(
            nc,
            kernel_size,
            ndf,
            nd_layers,
            norm_type)

    def forward(self, x: Tensor) -> Tuple[List, List, List]:
        """Weight-shared multi-resolution patch discriminator forward.

        Parameters:
        -----------
            `x`: input images.

        Returns:
        --------
            `output_list`:    list of output tensors produced by
                            individual patch discriminators.
            `features_list`:  list of lists of features produced by
                            individual patch discriminators.
            `input_list`:     list of downsampled input images.
        """
        input_list = []
        output_list = []
        features_list = []
        input_downsampled = x
        for _ in range(self.num_discriminators):
            input_list.append(input_downsampled)
            output, features = self.discriminator(input_downsampled)
            output_list.append(output)
            features_list.append(features)
            input_downsampled = nn.functional.interpolate(
                input_downsampled, scale_factor=0.5, mode='bilinear',
                align_corners=True)

        return output_list, features_list, input_list
        