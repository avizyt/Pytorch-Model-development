import torch
import torch.nn as nn
import torch.nn.functional as F

def CNNblock(in_features, out_features, *args, **kwarg):
    return nn.Sequential(
        nn.Conv2d(in_features, out_features, *args, **kwarg),
        nn.BatchNorm2d(out_features),
        nn.ReLU()
    )

def CNNdecoder(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.Sigmoid()
    )

class CNNClassifier():

    def __init__(self, in_features,encoder_size, decoder_size, n_class) -> None:
        super(CNNClassifier, self).__init__()

        self.encoder_size = [in_features, *encoder_size]
        self.decoder_size = [32 * 28 * 28, *decoder_size]

        CNN_blocks = [CNNblock(in_features, out_features, kernel_size=3, stride=1, padding=1) for in_features, out_features in zip(self.encoder_size, self.encoder_size[1:])]

        self.encoder = nn.Sequential(*CNN_blocks)

        decoder_block = [CNNdecoder(in_features, out_features) for in_features, out_features in zip(self.decoder_size, self.decoder_size[1:])] 

        self.decoder = nn.Sequential(*decoder_block)

        self.fc_last = nn.Linear(self.decoder_size[-1], n_class)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)

        return x



if __name__ == "__main__":
    model = CNNClassifier(1,[32,64,128,256],[1024,512], 10)
    print(model.encoder)