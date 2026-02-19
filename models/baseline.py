import torch
import torch.nn as nn
from models.resnet_block import ResidualBlock

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(ResidualBlock(32, 64), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(ResidualBlock(64, 128), nn.MaxPool2d(2))

        self.classifier = nn.Linear(128 * 4 *4, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    


        