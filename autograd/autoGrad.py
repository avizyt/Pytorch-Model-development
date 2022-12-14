import threading
import torch

def train_fn():
    x = torch.ones(5,5, requires_grad= True)
    y = (x+3) * (x+4) * 0.5
    y.sum().backward()


threads = []
for _ in range(10):
    p = threading.Thread(target=train_fn, args=())
    p.start()
    threads.append(p)
for p in threads:
    p.join()
