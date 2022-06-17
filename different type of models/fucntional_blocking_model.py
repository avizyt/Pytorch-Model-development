import torch
import torch.nn as nn
import torch.nn.functional as F

def CNNblock(in_features, out_features, *args, **kwarg):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, *args, **kwarg),
        nn.BatchNorm2d(out_features),
        nn.ReLU()
    )

class CNNClassifier(nn.Module):

    def __init__(self, in_features, n_class) -> None:
        super(CNNClassifier, self).__init__()
        self.conv_block1 = CNNblock(in_features, 32, kernel_size=3, stride=1, padding=1)
        self.conv_block2 = CNNblock(32,64, kernel_size=3, stride=1, padding=1)
        self.conv_block3 = CNNblock(64,128, kernel_size=3, stride=1, padding=1)

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



# more cleaner code

class CNNClassifier2():

    def __init__(self, in_features, n_class) -> None:
        super(CNNClassifier2, self).__init__()

        self.encoder = nn.Sequential(
            CNNblock(in_features, 32, kernel_size=3, stride=1, padding=1),
            CNNblock(32,64, kernel_size=3, stride=1, padding=1),
            CNNblock(64,128, kernel_size=3, stride=1, padding=1)
        )

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
    model = CNNClassifier2(1, 10)
    print(model.encoder)
    print("=========================================================")
    print(model.decoder)
