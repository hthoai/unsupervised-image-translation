import torch
import torch.nn as nn
from torch import Tensor

from lib.layers import ConvNormRelu, GaussianNoiseLayer
from lib.generators import ResidualGenerator


class SharedWeightResidualGenerator(ResidualGenerator):
    """Adapted from https://github.com/mingyuliutw/UNIT"""
    
    def __init__(
        self,
        nz: int,
        nc: int,
        ngf: int = 64,
        norm_type: str = "batch",
        ng_blocks: int = 6,
        n_downsampling: int = 2,
        n_enc_shared_blk: int = 1,
        n_gen_shared_blk: int = 1,
    ):
        super().__init__(nz, nc, ngf, norm_type, ng_blocks, n_downsampling)
        ############################
        # ENCODERS
        ############################
        self.encode_A = self.encoder
        self.encode_B = self.encoder
        ############################
        # SHARED LAYERS
        ############################
        _dim = list(self.encoder[-1].modules())[2].out_channels
        enc_shared = []
        for _ in range(0, n_enc_shared_blk):
            enc_shared += [
                ConvNormRelu(
                    _dim,
                    _dim,
                    conv_type="forward",
                    norm_type="instance",
                    padding=1,
                    leaky=0,
                )
            ]
        enc_shared += [GaussianNoiseLayer()]
        dec_shared = []
        for _ in range(0, n_gen_shared_blk):
            dec_shared += [
                ConvNormRelu(
                    _dim,
                    _dim,
                    conv_type="forward",
                    norm_type="instance",
                    padding=1,
                    leaky=0,
                )
            ]
        self.enc_shared = nn.Sequential(*enc_shared)
        self.dec_shared = nn.Sequential(*dec_shared)
        ############################
        # DECODERS
        ############################
        self.decode_A = self.decoder
        self.decode_B = self.decoder

    def forward(self, x_A: Tensor, x_B: Tensor):
        out = torch.cat((self.encode_A(x_A), self.encode_B(x_B)), 0)
        shared = self.enc_shared(out)
        out = self.dec_shared(shared)
        out_A = self.decode_A(out)
        out_B = self.decode_B(out)
        x_Aa, x_Ba = torch.split(out_A, x_A.size(0), dim=0)
        x_Ab, x_Bb = torch.split(out_B, x_A.size(0), dim=0)
        return x_Aa, x_Ba, x_Ab, x_Bb, shared

    def forward_a2b(self, x_A):
        out = self.encode_A(x_A)
        shared = self.enc_shared(out)
        out = self.dec_shared(shared)
        out = self.decode_B(out)
        return out, shared

    def forward_b2a(self, x_B):
        out = self.encode_B(x_B)
        shared = self.enc_shared(out)
        out = self.dec_shared(shared)
        out = self.decode_A(out)
        return out, shared
