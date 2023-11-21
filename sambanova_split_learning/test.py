from torch import nn
import torch
a = torch.randn(3,5)
b = torch.tensor([1,0,2])
loss = nn.CrossEntropyLoss()(a, b)
a = 1