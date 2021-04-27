from typing import Any
import yaml
import torch
import lib.models as models
import lib.datasets as datasets


class Config:
    def __init__(self, config_path: str):
        self.config = {}
        self.config_str = ""
        self.load(config_path)

    def load(self, path: str) -> None:
        with open(path, "r") as file:
            self.config_str = file.read()
        self.config = yaml.load(self.config_str, Loader=yaml.FullLoader)

    def get_dataset(self, data_split: str) -> None:
        return getattr(datasets, self.config["datasets"][data_split]["name"])(
            **self.config["datasets"][data_split]["parameters"]
        )

    def __repr__(self):
        return self.config_str

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config
