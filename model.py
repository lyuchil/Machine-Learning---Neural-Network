import torch
import torch.nn as nn
import torch.optim as optim

OUT_CHANNELS = 64
KERNAL_SIZE = 3


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.convOne = nn.Conv2d(6, OUT_CHANNELS, KERNAL_SIZE)
        self.convTwo = nn.Conv2d(OUT_CHANNELS, OUT_CHANNELS, KERNAL_SIZE) 
        self.convThree = nn.Conv2d(OUT_CHANNELS, OUT_CHANNELS, KERNAL_SIZE)
        self.fcOne = nn.Linear(OUT_CHANNELS * 8 * 8, 128)
        self.fcTwo = nn.Linear(OUT_CHANNELS * 8 * 8, 128)

    def forward(self, x):
        x = torch.relu(self.convOne(x))