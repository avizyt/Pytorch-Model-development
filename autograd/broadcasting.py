import torch

x = torch.empty(5,1,4,1)
y = torch.empty(3,1,1)
print((x+y).size())

xy = torch.add(torch.ones(4,1), torch.randn(4))
print(xy)
