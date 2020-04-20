from typing import Optional, Tuple

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.losses import multi_class_log_loss as criterion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(
    model: nn.Module,
    optimizer: optim.Optimizer,  # type: ignore
    num_epochs: int,
    trainloader: DataLoader,
    valloader: Optional[DataLoader] = None,
) -> Tuple[float, float]:
    """ Helper function that trains a model

    :param model: The model to train
    :param optimizer: The optimizer to use for training
    :param num_epochs: The number of epochs to train for
    :param trainloader: The DataLoader containing the training data
    :param valloader: The DataLoader containing the validation data, if None skip validation

    :return: The training and validation losses, if case of no validation the loss is 0
    """
    model.to(device)

    verbosity_level = 2
    train_loss = 0.0
    val_loss = 0.0

    print(f"Training the model for {num_epochs} epochs...")
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = F.softmax(model(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % verbosity_level == verbosity_level - 1:  # print every  mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / i))

        train_loss = running_loss / i
        print(f"[{epoch+1}/{num_epochs+1}] train loss: {train_loss:.3f}")
        if valloader:
            with torch.no_grad():
                running_loss = 0.0
                for i, data in enumerate(valloader, 0):
                    inputs, labels = data[0].to(device), data[1].to(device)

                    outputs = F.softmax(model(inputs))
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                val_loss = running_loss / i
                print(f"[{epoch + 1}/{num_epochs + 1}] val loss: {train_loss:.3f}")

    print("Finished Training")
    return train_loss, val_loss


def evaluate_model(model: nn.Module, dataloader: DataLoader) -> torch.Tensor:
    """ Helper function that evaluate model against a dataset
    :param model: The model to evaluate
    :param dataloader: The DataLoader containing the evaluation data

    :return: The multi-class logloss score
    """
    y_prob = torch.Tensor()
    y_true = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs = data[0].to(device)

            outputs = F.softmax(model(inputs))
            y_prob = torch.cat([y_prob, outputs.cpu()], 0)
            y_true += list(data[1].numpy())

    return criterion(y_prob, torch.Tensor(y_true).long()).detach()  # type: ignore
