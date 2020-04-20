from typing import Any

import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms


class StateFarmDataset(Dataset):
    """Distracted Driver Detection dataset."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        """
        Args:
            df: the dataframe containing the filepath and label for each image
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files = df[["filepath", "label"]]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: Any) -> Any:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files.iloc[idx, 0]
        image = io.imread(img_name)
        try:
            label = int(self.files.iloc[idx, 1])
        except:
            label = self.files.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label
