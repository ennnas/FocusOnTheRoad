from torch.nn import Module


def set_parameter_requires_grad(model: Module, requires: bool) -> None:
    """ Helper function that sets the requires_grad parameter

    :param model: the model instance to be modified
    :param requires: where to enable or not requires grad
    """
    for param in model.parameters():
        param.requires_grad = requires
