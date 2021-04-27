from typing import Any, Dict, List, Tuple
import random
import glob

from PIL import Image
import numpy as np
import imgaug.augmenters as iaa
from torch.functional import Tensor

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class UnalignedDataset(Dataset):
    """Ref: https://github.com/Lornatang/CycleGAN-PyTorch"""

    def __init__(
        self,
        root: str,
        img_size: List,
        normalize: bool = True,
        transforms: Any = None,
        transform_rate: float = 0.5,
    ) -> None:
        if transforms is not None:
            transformations = [
                getattr(iaa, trans["name"])(**trans["parameters"])
                for trans in transforms
            ]
        else:
            self.transformations = []
        self.normalize = normalize
        self.transform = iaa.Sequential(
            [
                iaa.Sometimes(then_list=transformations, p=transform_rate),
                iaa.Resize({"height": img_size[0], "width": img_size[1]}),
            ]
        )
        self.to_tensor = ToTensor()
        self.files_A = sorted(glob.glob(root + "A/*.*"))
        self.files_B = sorted(glob.glob(root + "B/*.*"))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        # Get images
        real_A = Image.open(self.files_A[index % len(self.files_A)])
        real_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        # Normalize
        if self.normalize:
            real_A = (real_A - IMAGENET_MEAN) / IMAGENET_STD
            real_B = (real_B - IMAGENET_MEAN) / IMAGENET_STD
        # Transform
        real_A = self.transform(image=real_A.copy())
        real_B = self.transform(image=real_B.copy())

        return self.to_tensor(real_A), self.to_tensor(real_B)

    def __len__(self) -> int:
        return max(len(self.files_A), len(self.files_B))
