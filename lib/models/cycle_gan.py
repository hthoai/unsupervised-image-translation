from typing import Tuple

from torch import Tensor
import torch.nn as nn

from utils.replay_buffer import ReplayBuffer
from lib.models import ResidualGenerator, PatchDiscriminator


class CycleGAN(nn.modules):
    """Defines CycleGAN model."""

    def __init__(
        self,
        nc: int,
        ngf: int,
        ndf: int,
        nz: int,
        ng_blocks: int,
        nd_layers: int,
        norm_layer: nn,
    ) -> None:
        super(CycleGAN, self).__init__()
        self.G_AB = ResidualGenerator(nz, nc, ngf, norm_layer, ng_blocks)
        self.G_BA = ResidualGenerator(nz, nc, ngf, norm_layer, ng_blocks)
        self.D_A = PatchDiscriminator(nc, ndf, nd_layers, norm_layer)
        self.D_B = PatchDiscriminator(nc, ndf, nd_layers, norm_layer)
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def forward(
        self, real_A: Tensor, real_B: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Translation
        fake_A = self.G_BA(real_B)
        fake_B = self.G_AB(real_A)
        # Back translation and reconciliation
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)

        return fake_A, fake_B, rec_A, rec_B
        