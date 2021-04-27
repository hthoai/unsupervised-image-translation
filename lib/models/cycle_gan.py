from typing import Any, Dict, List
import itertools

import torch
from torch import Tensor
import torch.nn as nn

from utils.replay_buffer import ReplayBuffer
from utils.lambda_lr import LambdaLR
from lib.models import ResidualGenerator, PatchDiscriminator


class CycleGAN(nn.Module):
    """Defines CycleGAN model."""

    def __init__(
        self,
        nc: int,
        nz: int,
        ngf: int,
        ndf: int,
        ng_blocks: int,
        nd_layers: int,
        norm_type: str,
        lambda_A: float,
        lambda_B: float,
        lambda_idt: float,
    ) -> None:
        """Construct CycleGAN.

        Parameters:
        -----------
            nc:          the number of image channels
            nz:          size of z latent vector
            ngf:         size of feature maps in generator
            ndf:         size of feature maps in discriminator
            ng_blocks:   the number of Residual blocks
            nd_layers:   the number of conv layers in the discriminator
            norm_type:   normalization layer type `batch` | `instance`
            lambda_A:    forward cycle loss weight
            lambda_B:    backward cycle loss weight
            lambda_idt:  identity loss weight
        """
        super(CycleGAN, self).__init__()
        # Generators
        self.G_AB = ResidualGenerator(nz, nc, ngf, norm_type, ng_blocks)
        self.G_BA = ResidualGenerator(nz, nc, ngf, norm_type, ng_blocks)
        # Discriminators
        self.D_A = PatchDiscriminator(nc, ndf, nd_layers, norm_type)
        self.D_B = PatchDiscriminator(nc, ndf, nd_layers, norm_type)
        # Relay Buffer
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()
        # Optimizers
        self.optimizers = []
        # Schedulers
        self.schedulers = []
        # Criterions
        self.gan_criterion = None
        self.cycle_criterion = None
        self.idt_criterion = None
        # Loss weights
        self.lambda_A = lambda_A
        self.lambda_B = lambda_B
        self.lambda_identity = lambda_idt

    def optimize_params(self, real_A: Tensor, real_B: Tensor) -> Dict:
        """Forward, backward, and optimize parameters.

        Parameters:
        -----------
            real_A: real image of domain A
            real_B: real image of domain B

        Returns:
        --------
            losses: {loss_G, loss_D, loss_cycle, loss_idt}
        """
        ############################
        # (I) Update G networks
        ############################
        self.optimizers[0].zero_grad()
        self.set_requires_grad([self.D_A, self.D_B], False)
        ## 1a) Translation
        fake_A = self.G_BA(real_B)
        fake_B = self.G_AB(real_A)
        ## 1b) Loss translation
        pred_fake_A = self.D_A(fake_A).view(-1)
        loss_gan_BA = self.gan_criterion(pred_fake_A, torch.zeros_like(pred_fake_A))
        pred_fake_B = self.D_B(fake_B).view(-1)
        loss_gan_AB = self.gan_criterion(pred_fake_B, torch.zeros_like(pred_fake_B))
        ## 2a) Back translation
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)
        ## 2b) Loss cycle
        loss_cycle_A = self.cycle_criterion(rec_A, real_A) * self.lambda_A
        loss_cycle_B = self.cycle_criterion(rec_B, real_B) * self.lambda_B
        ## 3a) Reconciliation
        same_A = self.G_AB(real_A)
        same_B = self.G_BA(real_B)
        ## 3b) Loss identity
        loss_idt_A = (
            self.idt_criterion(real_A, same_A) * self.lambda_A * self.lambda_identity
        )
        loss_idt_B = (
            self.idt_criterion(real_B, same_B) * self.lambda_B * self.lambda_identity
        )
        ## 4 Combine losses and calculate grads
        loss_G = (
            loss_gan_AB
            + loss_gan_BA
            + loss_cycle_A
            + loss_cycle_B
            + loss_idt_A
            + loss_idt_B
        )
        loss_G.backward()
        ## 5 Update Gs' weights
        self.optimizers[0].step()
        ############################
        # (II) Update D networks
        ############################
        self.optimizers[1].zero_grad()
        self.set_requires_grad([self.D_A, self.D_B], True)
        ## 1a) Loss D_A
        fake_A = self.fake_A_buffer(fake_A)
        pred_fake_A = self.D_A(fake_A)
        loss_fake_A = self.gan_criterion(pred_fake_A, torch.zeros_like(pred_fake_A))
        pred_real_A = self.D_A(real_A)
        loss_real_A = self.gan_criterion(pred_real_A, torch.ones_like(pred_real_A))
        ## 1b) Loss D_B
        fake_B = self.fake_B_buffer(fake_B)
        pred_fake_B = self.D_A(fake_B)
        loss_fake_B = self.gan_criterion(pred_fake_B, torch.zeros_like(pred_fake_B))
        pred_real_B = self.D_A(real_B)
        loss_real_B = self.gan_criterion(pred_real_B, torch.ones_like(pred_real_B))
        ## 2 Combine loss and calculate grads
        loss_D = (loss_fake_A + loss_real_A + loss_fake_B + loss_real_B) * 0.5
        loss_D.backward()
        ## 5 Update Ds' weights
        self.optimizers[1].step()

        losses = {
            "loss_G": loss_G,
            "loss_D": loss_D,
            "loss_cycle": loss_cycle_A + loss_cycle_B,
            "loss_idt": loss_idt_A + loss_idt_B,
        }

        return losses

    def set_optims_and_schedulers(self, cfgs: Any, starting_epoch: int) -> None:
        # Set optimizers
        G_params = itertools.chain(self.G_AB.parameters(), self.G_BA.parameters())
        G_optimizer = getattr(torch.optim, cfgs["optimizer"]["G"]["name"])(
            G_params, **cfgs["optimizer"]["G"]["parameters"]
        )
        self.optimizers.append(G_optimizer)
        D_params = itertools.chain(self.D_A.parameters(), self.D_B.parameters())
        D_optimizer = getattr(torch.optim, cfgs["optimizer"]["D"]["name"])(
            D_params, **cfgs["optimizer"]["D"]["parameters"]
        )
        self.optimizers.append(D_optimizer)
        # Set schedulers
        for optim in self.optimizers:
            self.schedulers.append(
                torch.optim.lr_scheduler.LambdaLR(
                    optim,
                    lr_lambda=LambdaLR(
                        cfgs["epochs"], starting_epoch, cfgs["decay_epoch"]
                    ).step,
                )
            )

    def set_criterions(self, cfgs: Any) -> None:
        self.gan_criterion = getattr(torch.nn, cfgs["criterion"]["gan"]["name"])(
            **cfgs["criterion"]["gan"]["parameters"]
        )
        self.cycle_criterion = getattr(torch.nn, cfgs["criterion"]["cycle"]["name"])(
            **cfgs["criterion"]["cycle"]["parameters"]
        )
        self.idt_criterion = getattr(torch.nn, cfgs["criterion"]["idt"]["name"])(
            **cfgs["criterion"]["idt"]["parameters"]
        )

    def set_requires_grad(self, nets: List, requires_grad: bool = False) -> None:
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
