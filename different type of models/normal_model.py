import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):

    def __init__(self, in_features, n_class) -> None:
        super(CNNClassifier, self).__init__()
        # Encoding
        self.conv1 = nn.Conv2d(in_features, 32, kernel_size=3, stride=1, padding=1 )
        self.batchN1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1 )
        self.batchN2 = nn.BatchNorm2d(32)

        # decoding
        self.fc1 = nn.Linear(32 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, n_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchN1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.batchN2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1) # flattening 

        x = self.fc1(x)
        x = F.Sigmoid(x)
        x = self.fc2(x)

        return x


model = CNNClassifier(1, 10)
print(model)

