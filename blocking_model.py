from matplotlib.lines import Line2D
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):

    def __init__(self, in_features, n_class) -> None:
        super(CNNClassifier, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_features, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(32 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_class)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.view (x.size(0), -1)

        x = self.decoder(x)

        return x


model = CNNClassifier(1, 10)
print(model)
