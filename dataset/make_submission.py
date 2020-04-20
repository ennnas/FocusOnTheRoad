import os
from typing import List, Tuple

import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.load_statefarm import get_dataloader

TEST_DIR = "./data/imgs/test"


def _get_submission_dataloader() -> DataLoader:
    """ Get the DataLoader for the submission data"""
    files = os.listdir(TEST_DIR)

    sub_df = pd.DataFrame(
        [(os.path.join(TEST_DIR, file), file) for file in files], columns=["filepath", "label"]
    )
    print(f"Loaded {len(sub_df)} submission files")
    return get_dataloader(sub_df, rotate=False)


def _make_predictions(
    model: torch.nn.Module, device: torch.device
) -> Tuple[torch.Tensor, List[str]]:
    """ Get the predictions from the model """
    target_images = []
    predictions = torch.Tensor()
    print(f"Starting prediction on submission dataset using {device}")

    dataloader = _get_submission_dataloader()
    with torch.no_grad():
        model.eval()
        for i, data in tqdm(enumerate(dataloader, 0)):
            inputs = data[0].to(device)

            out = F.softmax(model(inputs), dim=1)
            target_images += data[1]
            predictions = torch.cat([predictions, out.cpu()], 0)
    return predictions, target_images


def _save_predictions(predictions: torch.Tensor, filenames: List[str]) -> None:
    """ Create the submission dataframe containing the predictions"""
    prediction_df = pd.DataFrame(predictions.numpy(), columns=[f"c{i}" for i in range(10)])
    prediction_df.insert(0, "img", filenames)
    print(f"saving DataFrame of size {prediction_df.shape}...")
    prediction_df.to_csv("submission.csv", header=True, index=False)


def prepare_submission(model: torch.nn.Module, device: torch.device = torch.device("cpu")):
    """ Generate the csv submission file from the trained model"""
    predictions, filenames = _make_predictions(model, device)
    _save_predictions(predictions, filenames)
