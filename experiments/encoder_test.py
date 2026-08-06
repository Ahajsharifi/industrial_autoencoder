import torch

from models.encoder import Encoder


model = Encoder()

x = torch.randn(8, 3, 128, 128)

y = model(x)

print(y.shape)