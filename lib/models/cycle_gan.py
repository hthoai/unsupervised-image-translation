from typing import Any, Dict, List, Tuple
import itertools

import torch
from torch import Tensor
import torch.nn as nn

from utils.replay_buffer import ReplayBuffer
from utils.lambda_lr import LambdaLR
from utils.init_weight import init_weights
from lib.discriminators import PatchDiscriminator
from lib.generators import ResidualGenerator


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
        init_weights(self.G_AB)
        self.G_BA = ResidualGenerator(nz, nc, ngf, norm_type, ng_blocks)
        init_weights(self.G_BA)
        # Discriminators
        self.D_A = PatchDiscriminator(nc, ndf, nd_layers, norm_type)
        init_weights(self.D_A)
        self.D_B = PatchDiscriminator(nc, ndf, nd_layers, norm_type)
        init_weights(self.D_B)
        # Relay Buffer
        self.replay_buffer = {"fake_A": ReplayBuffer(), "fake_B": ReplayBuffer()}
        # Optimizers
        self.optimizers = {}
        # Schedulers
        self.schedulers = {}
        # Criterions
        self.criterions = {"gan": None, "cycle": None, "idt": None}
        # Loss weights
        self.lambdas = {"A": lambda_A, "B": lambda_B, "idt": lambda_idt}

    def optimize_params(
        self, real_A: Tensor, real_B: Tensor
    ) -> Tuple[Dict, Dict, Dict]:
        """Forward, backward, and optimize parameters.

        Parameters:
        -----------
            real_A: real image of domain A
            real_B: real image of domain B

        Returns:
        --------
            losses:   {loss_G, loss_D, loss_cycle, loss_idt}
            domain_A: {real, fake, rec, idt}
            domain_B: {real, fake, rec, idt}
        """
        ############################
        # (I) Update G networks
        ############################
        self.optimizers["G"].zero_grad()
        self.set_requires_grad([self.D_A, self.D_B], False)
        ## 1a) Translation
        fake_A = self.G_BA(real_B)
        fake_B = self.G_AB(real_A)
        ## 1b) Translation loss
        pred_fake_B = self.D_B(fake_B)
        loss_gan_AB = self.criterions["gan"](pred_fake_B, torch.ones_like(pred_fake_B))
        pred_fake_A = self.D_A(fake_A)
        loss_gan_BA = self.criterions["gan"](pred_fake_A, torch.ones_like(pred_fake_A))
        ## 2a) Back translation
        rec_A = self.G_BA(fake_B)
        rec_B = self.G_AB(fake_A)
        ## 2b) Back translation (cycle) loss
        loss_cycle_A = self.criterions["cycle"](rec_A, real_A) * self.lambdas["A"]
        loss_cycle_B = self.criterions["cycle"](rec_B, real_B) * self.lambdas["B"]
        ## 3a) Reconciliation (identity)
        idt_A = self.G_AB(real_B)
        idt_B = self.G_BA(real_A)
        ## 3b) Reconciliation (identity) loss
        loss_idt_A = (
            self.criterions["idt"](idt_A, real_A)
            * self.lambdas["A"]
            * self.lambdas["idt"]
        )
        loss_idt_B = (
            self.criterions["idt"](idt_B, real_B)
            * self.lambdas["B"]
            * self.lambdas["idt"]
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
        self.optimizers["G"].step()
        ############################
        # (II) Update D networks
        ############################
        self.set_requires_grad([self.D_A, self.D_B], True)
        ## D_A
        self.optimizers["D_A"].zero_grad()
        ### 1a) Loss D_A
        fake_A_buff = self.replay_buffer["fake_A"](fake_A)
        pred_fake_A = self.D_A(fake_A_buff.detach())
        loss_fake_A = self.criterions["gan"](pred_fake_A, torch.zeros_like(pred_fake_A))
        pred_real_A = self.D_A(real_A)
        loss_real_A = self.criterions["gan"](pred_real_A, torch.ones_like(pred_real_A))
        ### 1b) Backward D_A
        loss_DA = (loss_fake_A + loss_real_A) * 0.5
        loss_DA.backward()
        self.optimizers["D_A"].step()
        ## D_B
        self.optimizers["D_B"].zero_grad()
        ### 2a) Loss D_B
        fake_B_buff = self.replay_buffer["fake_B"](fake_B)
        pred_fake_B = self.D_B(fake_B_buff.detach())
        loss_fake_B = self.criterions["gan"](pred_fake_B, torch.zeros_like(pred_fake_B))
        pred_real_B = self.D_B(real_B)
        loss_real_B = self.criterions["gan"](pred_real_B, torch.ones_like(pred_real_B))
        ### 2b) Backward D_B
        loss_DB = (loss_fake_B + loss_real_B) * 0.5
        loss_DB.backward()
        ## 3 Update Ds' weights
        self.optimizers["D_B"].step()

        # Save for logging
        domain_A = {"real": real_A, "fake": fake_B, "rec": rec_A, "idt": idt_B}
        domain_B = {"real": real_B, "fake": fake_A, "rec": rec_B, "idt": idt_A}

        losses = {
            "A": {"G": (loss_gan_AB + loss_cycle_A + loss_idt_A) / 3, "D": loss_DA},
            "B": {"G": (loss_gan_BA + loss_cycle_B + loss_idt_B) / 3, "D": loss_DB},
        }

        return losses, domain_A, domain_B

    def set_optims_and_schedulers(self, cfgs: Any, starting_epoch: int) -> None:
        # Set optimizers
        # Gs
        G_params = itertools.chain(self.G_AB.parameters(), self.G_BA.parameters())
        G_optimizer = getattr(torch.optim, cfgs["optimizer"]["G"]["name"])(
            G_params, **cfgs["optimizer"]["G"]["parameters"]
        )
        self.optimizers["G"] = G_optimizer
        # D_A
        DA_optimizer = getattr(torch.optim, cfgs["optimizer"]["D"]["name"])(
            self.D_A.parameters(), **cfgs["optimizer"]["D"]["parameters"]
        )
        self.optimizers["D_A"] = DA_optimizer
        # D_B
        DB_optimizer = getattr(torch.optim, cfgs["optimizer"]["D"]["name"])(
            self.D_B.parameters(), **cfgs["optimizer"]["D"]["parameters"]
        )
        self.optimizers["D_B"] = DB_optimizer
        # Set schedulers
        for optim in self.optimizers.keys():
            self.schedulers[optim] = torch.optim.lr_scheduler.LambdaLR(
                self.optimizers[optim],
                lr_lambda=LambdaLR(
                    cfgs["epochs"], starting_epoch, cfgs["decay_epoch"]
                ).step,
            )

    def set_criterions(self, cfgs: Any) -> None:
        self.criterions["gan"] = getattr(torch.nn, cfgs["criterion"]["gan"]["name"])(
            **cfgs["criterion"]["gan"]["parameters"]
        )
        self.criterions["cycle"] = getattr(
            torch.nn, cfgs["criterion"]["cycle"]["name"]
        )(**cfgs["criterion"]["cycle"]["parameters"])
        self.criterions["idt"] = getattr(torch.nn, cfgs["criterion"]["idt"]["name"])(
            **cfgs["criterion"]["idt"]["parameters"]
        )

    def set_requires_grad(self, nets: List, requires_grad: bool = False) -> None:
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
