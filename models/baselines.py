from torch import nn
from torchvision.models import mobilenet_v2, vgg19_bn

from models.utils import set_parameter_requires_grad


class MobileNet(nn.Module):
    """ MobileNet v2 architecture, https://arxiv.org/abs/1801.04381"""

    def __init__(self, num_classes: int, pretrained: bool = True, lock_features: bool = False):
        super().__init__()
        # load pretrained models
        self.model = mobilenet_v2(pretrained=pretrained, progress=True)
        # if lock_features is True, don't require gradients for features extraction block
        if lock_features:
            set_parameter_requires_grad(self.model.features, False)
        # change the last layer to return the correct num_classes
        self.model.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):  # type: ignore
        return self.model(x)


class VGG19(nn.Module):
    """ VGG19 network with BatchNormalization, https://arxiv.org/pdf/1409.1556.pdf"""

    def __init__(self, num_classes: int, pretrained: bool = True, lock_features: bool = False):
        super().__init__()
        # load pretrained models
        self.model = vgg19_bn(pretrained=pretrained, progress=True)
        # if lock_features is True, don't require gradients for features extraction block
        if lock_features:
            set_parameter_requires_grad(self.model.features, False)
        # change the last layer to return the correct num_classes
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):  # type: ignore
        return self.model(x)
