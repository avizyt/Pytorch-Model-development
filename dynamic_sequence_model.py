import torch
import torch.nn as nn
import torch.nn.functional as F

def CNNblock(in_features, out_features, *args, **kwarg):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, *args, **kwarg),
        nn.BatchNorm2d(out_features),
        nn.ReLU()
    )

class CNNClassifier():

    def __init__(self, in_features, n_class) -> None:
        super(CNNClassifier, self).__init__()

        self.encoder_size = [in_features, 32, 64, 128, 256]

        CNN_blocks = [CNNblock(in_features, out_features, kernel_size=3, stride=1, padding=1) for in_features, out_features in zip(self.encoder_size, self.encoder_size[1:])]

        self.encoder = nn.Sequential(*CNN_blocks)

        self.decoder = nn.Sequential(
            nn.Linear(32 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_class)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x


class CNNClassifier_modified():

    def __init__(self, in_features,encoder_size, n_class) -> None:
        super(CNNClassifier_modified, self).__init__()

        self.encoder_size = [in_features, *encoder_size]

        CNN_blocks = [CNNblock(in_features, out_features, kernel_size=3, stride=1, padding=1) for in_features, out_features in zip(self.encoder_size, self.encoder_size[1:])]

        self.encoder = nn.Sequential(*CNN_blocks)

        self.decoder = nn.Sequential(
            nn.Linear(32 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_class)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x


if __name__ == "__main__":
    model = CNNClassifier_modified(1,[32,64,128,256], 10)
    print(model.encoder)