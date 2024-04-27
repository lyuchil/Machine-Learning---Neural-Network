import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time
import pandas as pd
from data import ChessDataset, printMove
from torch.utils.data import DataLoader, Dataset
FILENAME = "parsed_data_2018-06.csv"
FILENAMES = ["2023-10", "2023-11"]

# Hyperparameters
BATCH_SIZE = 64 # The batch size in moves
SHUFFLE_DATA = False
NUM_EPOCHS = 1
LEARNING_RATE = 0.00001 # 0.001 
OUT_CHANNELS = 64

KERNAL_SIZE = 3
PADDING = 1

START_TIME = time.time()

def timeSinceStart():
    return time.time() - START_TIME

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(18, OUT_CHANNELS, KERNAL_SIZE, padding=PADDING)
        self.conv2 = nn.Conv2d(OUT_CHANNELS, OUT_CHANNELS, KERNAL_SIZE, padding=PADDING) 
        self.conv3 = nn.Conv2d(OUT_CHANNELS, OUT_CHANNELS, KERNAL_SIZE, padding=PADDING)
        self.fc1 = nn.Linear(OUT_CHANNELS * 8 * 8, OUT_CHANNELS * 8 * 8) # 4099, 4096
        self.fc2 = nn.Linear(OUT_CHANNELS * 8 * 8, OUT_CHANNELS * 8 * 8)
        self.fc3 = nn.Linear(OUT_CHANNELS * 8 * 8, 128)

    def forward(self, x, metadata):
        # print(x.shape)
        x = torch.relu(self.conv1(x))
        # print(x.shape)
        x = torch.relu(self.conv2(x))
        # print(x.shape)
        x = torch.relu(self.conv3(x))
        # print(x.shape)
        x = x.view(x.shape[0], -1) # Flattens to feed into FC layer
        # print(x.shape, metadata.shape)
        # x = torch.cat((x, metadata), dim=1) # Appends metadata
        # print(x.shape)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

def train(dataset):
    # Data initialization
    train_data = dataset
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)

    # Model initialization
    model = Model()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training
    for epoch in range(NUM_EPOCHS):
        print(f"Started epoch {epoch+1} at time {timeSinceStart()}")
        model.train()
        for x, metadata, y in train_loader:
            optimizer.zero_grad()
            predictions = model(x, metadata)
            loss = loss_function(predictions, y)
            loss.backward()
            optimizer.step()
    
    print("Finished training at time", timeSinceStart())
    # x, metadata, y = train_data.__getitem__(0)

    x, metadata, y  = next(iter(train_loader))
    # printMove(x[:1])
    # print(y[:1].view(2,8,8))
    # print(x[:1].shape)
    print(metadata[:1])
    pred = model(x[:1], metadata[:1])
    print(pred.view(2, 8, 8))
    
if(__name__ == "__main__"):
    train_data = ChessDataset(FILENAMES)
    train(train_data)