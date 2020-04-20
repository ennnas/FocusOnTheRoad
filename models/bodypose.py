from collections import OrderedDict

import torch
import torch.nn as nn

from models.utils import make_layers


class BodyPose(nn.Module):
    """
    The Pytorch implementation of the BodyPose model described in the first version of the
    Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields, a.k.a OpenPose
    https://arxiv.org/pdf/1611.08050.pdf
    """

    def __init__(self):
        super().__init__()
        # This model uses a “two-branch multi-stage” CNN
        # Two branch means that the CNN produces two different outputs.
        # Multi-stage means that the network is stacked one on top of the other at every stage

        # these layers have no relu layer
        no_relu_layers = [
            "conv5_5_CPM_L1",
            "conv5_5_CPM_L2",
            "Mconv7_stage2_L1",
            "Mconv7_stage2_L2",
            "Mconv7_stage3_L1",
            "Mconv7_stage3_L2",
            "Mconv7_stage4_L1",
            "Mconv7_stage4_L2",
            "Mconv7_stage5_L1",
            "Mconv7_stage5_L2",
            "Mconv7_stage6_L1",
            "Mconv7_stage6_L1",
        ]
        blocks = {}

        # CNN layers of VGG-19
        block0 = OrderedDict(
            [
                ("conv1_1", [3, 64, 3, 1, 1]),
                ("conv1_2", [64, 64, 3, 1, 1]),
                ("pool1_stage1", [2, 2, 0]),
                ("conv2_1", [64, 128, 3, 1, 1]),
                ("conv2_2", [128, 128, 3, 1, 1]),
                ("pool2_stage1", [2, 2, 0]),
                ("conv3_1", [128, 256, 3, 1, 1]),
                ("conv3_2", [256, 256, 3, 1, 1]),
                ("conv3_3", [256, 256, 3, 1, 1]),
                ("conv3_4", [256, 256, 3, 1, 1]),
                ("pool3_stage1", [2, 2, 0]),
                ("conv4_1", [256, 512, 3, 1, 1]),
                ("conv4_2", [512, 512, 3, 1, 1]),
                ("conv4_3_CPM", [512, 256, 3, 1, 1]),
                ("conv4_4_CPM", [256, 128, 3, 1, 1]),
            ]
        )

        # Stage 1 - the network produces an initial set of
        #   - part affinity fields (PAFs) L
        #   - detection confidence maps S

        # Below we use the notation blockX_1 to define blocks that belong to the first branch
        # which predict the PAFs that represent a degree of association between different body parts
        block1_1 = OrderedDict(
            [
                ("conv5_1_CPM_L1", [128, 128, 3, 1, 1]),
                ("conv5_2_CPM_L1", [128, 128, 3, 1, 1]),
                ("conv5_3_CPM_L1", [128, 128, 3, 1, 1]),
                ("conv5_4_CPM_L1", [128, 512, 1, 1, 0]),
                ("conv5_5_CPM_L1", [512, 38, 1, 1, 0]),
            ]
        )

        # Below we use the notation blockX_2 to define blocks that belong to the second branch
        # which predict the confidence maps of different body parts location
        block1_2 = OrderedDict(
            [
                ("conv5_1_CPM_L2", [128, 128, 3, 1, 1]),
                ("conv5_2_CPM_L2", [128, 128, 3, 1, 1]),
                ("conv5_3_CPM_L2", [128, 128, 3, 1, 1]),
                ("conv5_4_CPM_L2", [128, 512, 1, 1, 0]),
                ("conv5_5_CPM_L2", [512, 19, 1, 1, 0]),
            ]
        )
        blocks["block1_1"] = block1_1
        blocks["block1_2"] = block1_2

        self.model0 = make_layers(block0, no_relu_layers)

        # Stages 2 - 6
        # predictions from previous stages are iteratively refined
        for i in range(2, 7):
            blocks[f"block{i}_1"] = OrderedDict(
                [
                    (f"Mconv1_stage{i}_L1", [185, 128, 7, 1, 3]),
                    (f"Mconv2_stage{i}_L1", [128, 128, 7, 1, 3]),
                    (f"Mconv3_stage{i}_L1", [128, 128, 7, 1, 3]),
                    (f"Mconv4_stage{i}_L1", [128, 128, 7, 1, 3]),
                    (f"Mconv5_stage{i}_L1", [128, 128, 7, 1, 3]),
                    (f"Mconv6_stage{i}_L1", [128, 128, 1, 1, 0]),
                    (f"Mconv7_stage{i}_L1", [128, 38, 1, 1, 0]),
                ]
            )

            blocks[f"block{i}_2"] = OrderedDict(
                [
                    (f"Mconv1_stage{i}_L2", [185, 128, 7, 1, 3]),
                    (f"Mconv2_stage{i}_L2", [128, 128, 7, 1, 3]),
                    (f"Mconv3_stage{i}_L2", [128, 128, 7, 1, 3]),
                    (f"Mconv4_stage{i}_L2", [128, 128, 7, 1, 3]),
                    (f"Mconv5_stage{i}_L2", [128, 128, 7, 1, 3]),
                    (f"Mconv6_stage{i}_L2", [128, 128, 1, 1, 0]),
                    (f"Mconv7_stage{i}_L2", [128, 19, 1, 1, 0]),
                ]
            )

        for k in blocks.keys():
            blocks[k] = make_layers(blocks[k], no_relu_layers)

        # all the stages that belong to the first branch
        self.model1_1 = blocks["block1_1"]
        self.model2_1 = blocks["block2_1"]
        self.model3_1 = blocks["block3_1"]
        self.model4_1 = blocks["block4_1"]
        self.model5_1 = blocks["block5_1"]
        self.model6_1 = blocks["block6_1"]

        # all the stages that belong to the second branch
        self.model1_2 = blocks["block1_2"]
        self.model2_2 = blocks["block2_2"]
        self.model3_2 = blocks["block3_2"]
        self.model4_2 = blocks["block4_2"]
        self.model5_2 = blocks["block5_2"]
        self.model6_2 = blocks["block6_2"]

    def forward(self, x):
        """ The forward method of the model.
        :param x: input torch.Tensor representing the original image

        :return: Tuple with the two torch.Tensor (PAFs L, confidence map S)
        """
        # feature map generated by the finetuned VGG-19
        F = self.model0(x)

        # initial set of confidence maps S and PAFs L
        out1_1 = self.model1_1(F)
        out1_2 = self.model1_2(F)
        out2 = torch.cat([out1_1, out1_2, F], 1)

        # Stage 2-6
        # predictions from both branches in the previous stage, along with the original image
        # features F, are concatenated and used to produce more refined predictions
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, F], 1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, F], 1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, F], 1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, F], 1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)
        return out6_1, out6_2
