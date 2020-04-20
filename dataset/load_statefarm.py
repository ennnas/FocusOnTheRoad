from typing import Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.dataset import StateFarmDataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_dataloader(
    df: pd.DataFrame, batch_size: int = 32, normalize: Optional[Tuple] = None, rotate: bool = True
) -> DataLoader:
    """ Helper function that build a DataLoader from a pandas DataFrame

    :param df: the pandas DataFrame containing the columns `filepath` and `label`
    :param batch_size: the batch size of the DataLoader
    :param normalize: A tuple with (mean, std) to normalize the data,
                      if None ImageNet normalization is applied
    :param rotate: if set to True images may be applied a random rotation of [-20,20]

    :return: the DataLoader built from the dataframe which returns (image, label) pairs
    """

    if normalize:
        normalization = transforms.Normalize(mean=normalize[0], std=normalize[1])
    else:
        normalization = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    transformations = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            normalization,
        ]
    )
    # for test dataframe we dont want to rotate the images
    if not rotate:
        print(f"Removing transformation {transformations.transforms.pop(2)}")

    dataset = StateFarmDataset(df=df, transform=transformations)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
