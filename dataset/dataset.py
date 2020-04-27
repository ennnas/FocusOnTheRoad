from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

from models.utils import padRightDownCorner


class StateFarmDataset(Dataset):
    """Distracted Driver Detection dataset."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        """ Initialize the dataset object

        :param df: the dataframe containing the filepath and label for each image
        :param transform: Optional transform to be applied on a sample.
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

        return image, label, img_name


class OpenPoseDataset(Dataset):
    """Distracted Driver Detection dataset used for keypoints extraction."""

    def __init__(self, df: pd.DataFrame):
        """
        :param df: the dataframe containing the filepath and label for each image
        """
        self.image_shape = (224, 298)  # width and height, i.e. cols, rows
        self.scale_search = 0.5
        self.boxsize = 368
        self.stride = 8
        self.padValue = 128
        self.scale = self.scale_search * self.boxsize / self.image_shape[0]

        self.files = df[["filepath", "label"]]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: Any) -> Any:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files.iloc[idx, 0]
        orig_img = cv2.imread(img_name)
        # resize the image to ease computation
        h = self.image_shape[0]
        w = self.image_shape[1]
        orig_img = cv2.resize(orig_img, dsize=(w, h))
        try:
            label = int(self.files.iloc[idx, 1])
        except:
            label = self.files.iloc[idx, 1]

        imageToTest = cv2.resize(
            orig_img, (0, 0), fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC
        )
        imageToTest_padded, pad = padRightDownCorner(imageToTest, self.stride, self.padValue)
        # openpose requires the image to be normalized in [-0.5, 0.5]
        im = np.transpose(np.float32(imageToTest_padded), (2, 0, 1)) / 256 - 0.5
        im = np.ascontiguousarray(im)

        image = torch.from_numpy(im).float()

        return image, label, pad, img_name
