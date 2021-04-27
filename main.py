import logging
import argparse

import torch

from lib.config import Config
from lib.runner import Runner
from lib.experiment import Experiment


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unsupervised Image-to-Image Translation"
    )
    parser.add_argument("mode", choices=["train", "test"], help="train|test")
    parser.add_argument("--exp_name", default="cycle_gan", help="Experiment name")
    parser.add_argument("--cfg", required=True, help="Config path")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--epoch", type=int, help="Epoch to test the model on")

    args = parser.parse_args()
    if args.resume and args.mode == "test":
        raise Exception("args.resume is set on `test` mode: can't resume testing")
    if args.epoch is not None and args.mode == "train":
        raise Exception("The `epoch` parameter should not be set when training")

    return args


def main() -> None:
    args = parse_args()
    exp = Experiment(args.exp_name, args, mode=args.mode)
    if args.cfg is None:
        cfg_path = exp.cfg_path
    else:
        cfg_path = args.cfg
    cfg = Config(cfg_path)
    exp.set_cfg(cfg, override=False)
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    )
    runner = Runner(cfg, exp, device, resume=args.resume)
    if args.mode == "train":
        try:
            runner.train()
        except KeyboardInterrupt:
            logging.info("Training interrupted.")
    # TO-DO
    # else:
    #     runner.test(epoch=args.epoch or exp.get_last_checkpoint_epoch())


if __name__ == "__main__":
    main()
