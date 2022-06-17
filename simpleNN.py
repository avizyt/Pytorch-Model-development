import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms

from torch import nn
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter


training_data = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

test_data = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=False,
    transform=transforms.ToTensor()
)

batch_size = 32

train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

for X, y in test_loader:
    print(f"Shape of X [N, C, H, W,] = {X.shape}")
    print(f"Shape of y [N, 1] = {y.shape} {y.dtype}")
    break

classes = (
    'T-shirt/top', 
    'Trouser', 
    'Pullover', 
    'Dress', 
    'Coat',
    'Sandal', 
    'Shirt', 
    'Sneaker', 
    'Bag', 
    'Ankle Boot'
    )

def show_image(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img/2 + 0.5
    img_arr = img.numpy()
    if one_channel:
        plt.imshow(img_arr, cmap="Greys")
    else:
        plt.imshow(np.transpose(img_arr, (1,2,0)))


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class SimpleNN(nn.Module):
    def __init__(self) -> None:
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),

        )

    def forward(self, X):
        X = self.flatten(X)
        logits = self.linear_relu_stack(X)
        return logits

model = SimpleNN().to(device=device)
print(model)        

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# tensorboard setup

# log_dir = "./runs/fashion_mnist_expt1"
# writer = SummaryWriter(log_dir=log_dir)

# dataiter = iter(train_loader)
# images, labels = dataiter.next()

# img_grid = torchvision.utils.make_grid(images)

# show_image(img_grid, one_channel=True)

# writer.add_image("32 Fashion images", img_grid)




def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, cur = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{cur:>5d} / {size:>5d}")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# writer.add_graph(SimpleNN, images)
# writer.close()
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------")
    train(train_loader, model=model, loss_fn=loss_fn, optimizer=optimizer)
    test(test_loader, model=model, loss_fn=loss_fn)
print("You have high patience!!")
print("Relax it is Done!!!")


