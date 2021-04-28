import random
import logging
from typing import Any

import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader

import lib.models as models
from lib.config import Config
from lib.experiment import Experiment


class Runner:
    def __init__(
        self,
        cfg: Config,
        exp: Experiment,
        device: torch.device,
        resume: bool = False,
        deterministic: bool = False,
    ):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.logger = logging.getLogger(__name__)
        self.iters = 0

        # Fix seeds
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self) -> None:
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        if self.resume:
            starting_epoch += self.exp.get_last_checkpoint_epoch()

        model = self.get_model()
        model.set_optims_and_schedulers(self.cfg, starting_epoch)
        model.set_criterions(self.cfg)
        model = model.to(self.device)

        if self.resume:
            model = self.exp.load_last_train_state(model)
        max_epochs = self.cfg["epochs"]
        train_loader = self.get_data_loader(
            split="train", batch_size=self.cfg["batch_size"]
        )

        for epoch in trange(
            starting_epoch, max_epochs + 1, initial=starting_epoch, total=max_epochs
        ):
            self.exp.epoch_start_callback(epoch, max_epochs)
            pbar = tqdm(train_loader)
            model.train()

            for idx, (real_A, real_B) in enumerate(pbar):
                # Load to GPU
                real_A = real_A.to(self.device)
                real_B = real_B.to(self.device)
                # Forward, backward, and optimize params
                losses, domain_A, domain_B = model.optimize_params(real_A, real_B)

                # Log to progressing bar
                loss_components = losses["A"] | losses["B"]
                postfix_dict = {
                    key: float(value) for key, value in loss_components.items()
                }
                lr = model.optimizers["G"].param_groups[0]["lr"]
                self.exp.iter_end_callback(
                    epoch, max_epochs, idx, len(train_loader), losses, lr
                )
                pbar.set_postfix(ordered_dict=postfix_dict)
                self.iters += 1
                # Log image to tensorboard:
                if self.iters % self.cfg["log_image_interval"] == 0:
                    self.exp.log_image_and_hist_callback(
                        model, domain_A, domain_B, epoch, idx, len(train_loader)
                    )
                # TO-DO val step here

            # Update learning rate
            for scheduler in model.schedulers.keys():
                model.schedulers[scheduler].step()
            self.exp.epoch_end_callback(epoch, max_epochs, model)

        self.exp.train_end_callback()

    def get_model(self, **kwargs) -> Any:
        name = self.cfg["model"]["name"]
        parameters = self.cfg["model"]["parameters"]
        return getattr(models, name)(**parameters, **kwargs)

    def get_data_loader(
        self,
        split: str,
        batch_size: int = 1,
        shuffle: bool = False,
    ) -> DataLoader:
        """Returns torch.utils.data.DataLoader for custom dataset."""
        dataset = self.cfg.get_dataset(split)
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn_,
        )
        return data_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
