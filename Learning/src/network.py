import torch
from torch import nn
import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        # Feature extraction using MobileNet V3
        mobilenet = models.mobilenet.mobilenet_v3_small(pretrained=True)
        # Remove the last layer to keep only feature extraction
        mobilenet.classifier = Identity()

        self.feature_extractor = mobilenet
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(128, 2)

    def forward(self, x):
        features = self.feature_extractor(x)
        y = self.linear_relu_stack(features)
        y = self.output_layer(y)
        return y
