import os
import re
import logging
import subprocess
from typing import Any, Dict, List, Tuple

from collections import OrderedDict

from lib.config import Config

import torch
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter


class Experiment:
    def __init__(
        self,
        exp_name: str,
        args: Any = None,
        mode: str = "train",
        exps_basedir: str = "experiments",
        tensorboard_dir: str = "tensorboard",
    ):
        self.name = exp_name
        self.exp_dirpath = os.path.join(exps_basedir, exp_name)
        self.models_dirpath = os.path.join(self.exp_dirpath, "models")
        self.cfg_path = os.path.join(self.exp_dirpath, "config.yaml")
        self.code_state_path = os.path.join(self.exp_dirpath, "code_state.txt")
        self.log_path = os.path.join(self.exp_dirpath, "log_{}.txt".format(mode))
        self.results_path = os.path.join(
            self.exp_dirpath, "results_{}.csv".format(exp_name)
        )
        self.tensorboard_writer = SummaryWriter(os.path.join(tensorboard_dir, exp_name))
        self.cfg = None
        self.setup_exp_dir()
        self.setup_logging()

        if args is not None:
            self.log_args(args)

    def setup_exp_dir(self) -> None:
        if not os.path.exists(self.exp_dirpath):
            os.makedirs(self.exp_dirpath)
            os.makedirs(self.models_dirpath)
            self.save_code_state()

    def save_code_state(self) -> None:
        state = "Git hash: {}".format(
            subprocess.run(
                ["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, check=False
            ).stdout.decode("utf-8")
        )
        state += "\n*************\nGit diff:\n*************\n"
        state += subprocess.run(
            ["git", "diff"], stdout=subprocess.PIPE, check=False
        ).stdout.decode("utf-8")
        with open(self.code_state_path, "w") as code_state_file:
            code_state_file.write(state)

    def setup_logging(self) -> None:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
        )
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logging.basicConfig(
            level=logging.DEBUG, handlers=[file_handler, stream_handler]
        )
        self.logger = logging.getLogger(__name__)

    def log_args(self, args: Any) -> None:
        self.logger.debug("CLI Args:\n %s", str(args))

    def set_cfg(self, cfg: Config, override: bool = False) -> None:
        assert "model_checkpoint_interval" in cfg
        self.cfg = cfg
        if not os.path.exists(self.cfg_path) or override:
            with open(self.cfg_path, "w") as cfg_file:
                cfg_file.write(str(cfg))

    def get_last_checkpoint_epoch(self) -> int:
        pattern = re.compile("model_(\\d+).pt")
        last_epoch = -1
        for ckpt_file in os.listdir(self.models_dirpath):
            result = pattern.match(ckpt_file)
            if result is not None:
                epoch = int(result.groups()[0])
                if epoch > last_epoch:
                    last_epoch = epoch

        return last_epoch

    def get_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.models_dirpath, "model_{:04d}.pt".format(epoch))

    def get_epoch_model(self, epoch: int) -> Any:
        return torch.load(self.get_checkpoint_path(epoch))["model"]

    def load_last_train_state(self, model) -> Any:
        epoch = self.get_last_checkpoint_epoch()
        train_state_path = self.get_checkpoint_path(epoch)
        train_state = torch.load(train_state_path)
        model.load_state_dict(train_state["model"])

        return model

    def save_train_state(self, epoch: int, model: Any) -> None:
        train_state_path = self.get_checkpoint_path(epoch)
        torch.save(
            {"epoch": epoch, "model": model.state_dict()},
            train_state_path,
        )

    def iter_end_callback(
        self,
        epoch: int,
        max_epochs: int,
        iter_nb: int,
        max_iter: int,
        losses: Dict,
        lr: float,
    ) -> None:
        line = "Epoch [{}/{}] - Iter [{}/{}] - ".format(
            epoch, max_epochs, iter_nb, max_iter
        )
        for key in losses:
            line += " - ".join(
                [
                    "{}: {:.5f}".format(component, losses[key][component])
                    for component in losses[key]
                ]
            )
        self.logger.debug(line)
        overall_iter = ((epoch - 1) * max_iter) + iter_nb
        for key in losses:
            self.tensorboard_writer.add_scalars(
                "loss/{}".format(key), losses[key], overall_iter
            )
        self.tensorboard_writer.add_scalar("lr", lr, overall_iter)

    def log_image_and_hist_callback(
        self, model: Any, domain_A: Dict, domain_B: Dict, epoch, iter_nb, max_iter
    ) -> None:
        assert self.cfg["num_image_log"] <= self.cfg["batch_size"]
        overall_iter = ((epoch - 1) * max_iter) + iter_nb
        n_imgs = self.cfg["num_image_log"]
        img_shape = domain_A["real"].data[0].shape
        ############################
        # Domain A
        ############################
        images_A = torch.stack(
            (
                domain_A["real"].data[:n_imgs],
                domain_A["fake"].data[:n_imgs],
                domain_A["rec"].data[:n_imgs],
                domain_A["idt"].data[:n_imgs],
            ),
            dim=1,
        ).view(len(domain_A) * n_imgs, img_shape[0], img_shape[1], img_shape[2])
        # Write to tensorboard
        ## Image
        self.tensorboard_writer.add_image(
            self.cfg["task"][0] + " (real | fake | rec | idt)",
            vutils.make_grid(
                images_A, nrow=len(domain_A), normalize=True, scale_each=True
            ),
            overall_iter,
        )
        ## Histogram
        self.tensorboard_writer.add_histogram(
            self.cfg["task"][0] + "/real", model.D_A(domain_A["real"]), overall_iter
        )
        self.tensorboard_writer.add_histogram(
            self.cfg["task"][0] + "/fake", model.D_A(domain_A["fake"]), overall_iter
        )
        ############################
        # Domain B
        ############################
        images_B = torch.stack(
            (
                domain_B["real"].data[:n_imgs],
                domain_B["fake"].data[:n_imgs],
                domain_B["rec"].data[:n_imgs],
                domain_B["idt"].data[:n_imgs],
            ),
            dim=1,
        ).view(len(domain_B) * n_imgs, img_shape[0], img_shape[1], img_shape[2])
        # Write to tensorboard
        ## Image
        self.tensorboard_writer.add_image(
            self.cfg["task"][1] + " (real | fake | rec | idt)",
            vutils.make_grid(
                images_B, nrow=len(domain_B), normalize=True, scale_each=True
            ),
            overall_iter,
        )
        ## Histogram
        self.tensorboard_writer.add_histogram(
            self.cfg["task"][1] + "/real", model.D_B(domain_B["real"]), overall_iter
        )
        self.tensorboard_writer.add_histogram(
            self.cfg["task"][1] + "/fake", model.D_B(domain_B["fake"]), overall_iter
        )

    def epoch_start_callback(self, epoch: int, max_epochs: int) -> None:
        self.logger.debug(f"Epoch [{epoch}/{max_epochs}] starting.")

    def epoch_end_callback(
        self,
        epoch: int,
        max_epochs: int,
        model: Any,
    ) -> None:
        self.logger.debug(f"Epoch [{epoch}/{max_epochs}] finished.")
        if epoch % self.cfg["model_checkpoint_interval"] == 0:
            self.save_train_state(epoch, model)

    def train_start_callback(self, cfg: Config) -> None:
        self.logger.debug(f"Beginning training session. CFG used: {cfg}")

    def train_end_callback(self) -> None:
        self.logger.debug("Training session finished.")
