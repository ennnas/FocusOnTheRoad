import torch


def multi_class_log_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Multi-class log loss as defined in the kaggle competition
    https://www.kaggle.com/c/state-farm-distracted-driver-detection/overview/evaluation

    :param outputs: torch tensor containing the model predictions
    :param labels: torch tensor containing the truth labels

    :return: the loss score
    """
    loss = -torch.mean(torch.log(torch.gather(outputs, 1, labels.view(-1, 1))))

    return loss
