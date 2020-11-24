import torch
import torch.utils.data
from torchvision import transforms, datasets
import numpy as np


class AutoDriveGRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size * batch_size, 4)

    def forward(self, x, h):
        output, hidden = self.gru(x, h)
        output = output[:, -1, :].view(1, -1)
        output = self.fc(output)

        return output, hidden


c = AutoDriveGRU(10, 20, 2, 3)
a = torch.rand(3, 10, 10)
h = torch.rand(2, 3, 20)
a, h, = c(a, h)
print(a.shape)
a = a.squeeze()
print(a.shape)
a = [(x > 0).item() for x in a]
print(a)
a = [1 if x else 0 for x in a]
print(a)

print(h.shape)