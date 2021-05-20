from typing import Any, Dict, List, Tuple
import random
import glob

from PIL import Image
import numpy as np
from torch.functional import Tensor

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class UnalignedDataset(Dataset):
    """Ref: https://github.com/Lornatang/CycleGAN-PyTorch"""

    def __init__(self, root: str, img_size: List, transformations: Any = None) -> None:
        transforms_list = [transforms.ToTensor()]
        if transformations is not None:
            transforms_list += [
                getattr(transforms, trans["name"])(**trans["parameters"])
                for trans in transformations
            ]
        transforms_list += [
            transforms.Resize(size=img_size, interpolation=Image.BICUBIC)
        ]
        self.transform = transforms.Compose(transforms_list)
        self.files_A = sorted(glob.glob(root + "A/*.*"))
        self.files_B = sorted(glob.glob(root + "B/*.*"))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # Get images
        real_A = Image.open(self.files_A[index % len(self.files_A)]).convert("RGB")
        real_B = Image.open(
            self.files_B[random.randint(0, len(self.files_B) - 1)]
        ).convert("RGB")
        # Transform
        real_A = self.transform(real_A)
        real_B = self.transform(real_B)

        return real_A, real_B

    def __len__(self) -> int:
        return max(len(self.files_A), len(self.files_B))
