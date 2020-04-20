import argparse
from typing import Sequence

import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

from dataset.load_statefarm import get_dataloader
from dataset.model_selection import train_test_split
from models.baselines import VGG19
from models.trainer import evaluate_model, train_model

DATA_DIR = "data/"
CLASSES = {
    0: "safe driving",
    1: "texting - right",
    2: "talking on the phone - right",
    3: "texting - left",
    4: "talking on the phone - left",
    5: "operating the radio",
    6: "drinking",
    7: "reaching behind",
    8: "hair and makeup",
    9: "talking to passenger",
}


def get_dataframes(train_size: int = 600, val_size: int = 100) -> Sequence[pd.DataFrame]:
    """ Read the data and split images into train, validation and test sets
    :param train_size: The number of images per class to use for training
    :param val_size: The number of images per class to use for validation

    :return: the train, validation and test dataframes"""
    # read the drivers list from the csv
    df = pd.read_csv(DATA_DIR + "driver_imgs_list.csv")

    # compose the filepath and extract the label from the classname cX
    df["filepath"] = DATA_DIR + "imgs/train/" + df["classname"] + "/" + df["img"]
    df["label"] = df.classname.str[-1].astype(int)
    df["classname"] = df.label.map(CLASSES)

    # extract the training and validation set and use the remaining for testing
    train_df, val_df = train_test_split(
        df, train_size + val_size, val_size, group_ids=["classname"]
    )
    test_df = pd.concat([df, train_df, val_df]).drop_duplicates(keep=False)
    return train_df, val_df, test_df


def train(
    model: torch.nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    optimize_namer: str = "sgd",
    lr: float = 1e-3,
) -> None:
    """ Train and evaluate a VGG19 baseline

    :param model: The model to train
    :param trainloader: DataLoader used for training
    :param valloader: DataLoader used for validation
    :param optimize_namer: The optimizer used to optimize the model
    :param lr: The learning rate of the optimizer
    """
    # define the optimizer
    if optimize_namer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if optimize_namer == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)  # type: ignore

    # train the model
    train_loss, val_loss = train_model(
        model=model,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        trainloader=trainloader,
        valloader=valloader,
    )
    print(f"Final loss: train {round(train_loss, 3)}, validation {round(val_loss, 3)}")


def main(args: argparse.Namespace) -> None:
    # load the data
    print("Loading the data...")
    train_df, val_df, test_df = get_dataframes(args.train_size, args.val_size)
    print(f"Training set: {len(train_df)}\nValidation set: {len(val_df)}\nTest set: {len(test_df)}")
    # define the dataloaders
    trainloader = get_dataloader(train_df)
    valloader = get_dataloader(val_df)
    testloader = get_dataloader(test_df)

    # init the model
    print("Initializing the VGG19 model...")
    model = VGG19(num_classes=10, pretrained=True, lock_features=False)

    # train the model
    train(model, trainloader, valloader, args.optimizer, args.lr)

    # evaluate the model on the remaining samples
    test_loss = evaluate_model(model, testloader)
    print(f"VGG19 scored {test_loss} on a dataset of size {test_df.shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a baseline model on the StateFarmDataset.")
    parser.add_argument(
        "--num-epochs", type=int, default=5, help="the number of epochs to train the model"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="the batch size for the DataLoader"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adagrad",
        choices=["sgd", "adagrad"],
        help="the optimizer to use for training",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="the learning rate for the optimizer"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=32,
        help="the number of images per class to use for training",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=32,
        help="the number of images per class to use for validation",
    )

    args = parser.parse_args()
    main(args)
