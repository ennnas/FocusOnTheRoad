import os
from typing import List, Tuple

import pandas as pd
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.load_statefarm import get_dataloader

TEST_DIR = "./data/imgs/test"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _get_submission_dataloader() -> DataLoader:
    """ Get the DataLoader for the submission data"""
    files = os.listdir(TEST_DIR)

    sub_df = pd.DataFrame(
        [(os.path.join(TEST_DIR, file), file) for file in files], columns=["filepath", "label"]
    )
    return get_dataloader(sub_df, rotate=False)


def _make_predictions(
    model: torch.nn.Module, dataloader: DataLoader
) -> Tuple[torch.Tensor, List[str]]:
    """ Get the predictions from the model """
    target_images = []
    predictions = torch.Tensor()
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader, 0)):
            inputs = data[0].to(device)

            out = model(inputs)
            target_images += data[1]
            predictions = torch.cat([predictions, out.cpu()], 0)
    return predictions, target_images


def _save_predictions(predictions: torch.Tensor, filenames: List[str]) -> None:
    """ Create the submission dataframe containing the predictions"""
    prediction_df = pd.DataFrame(
        softmax(predictions.numpy(), axis=1), columns=[f"c{i}" for i in range(10)]
    )
    prediction_df.insert(0, "img", filenames)
    print(f"saving DataFrame of size {prediction_df.shape}...")
    prediction_df.to_csv("submission.csv", header=True, index=False)


def prepare_submission(model: torch.nn.Module):
    """ Generate the csv submission file from the trained model"""
    dataloader = _get_submission_dataloader()
    predictions, filenames = _make_predictions(model, dataloader)
    _save_predictions(predictions, filenames)
