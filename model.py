import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time
import pandas as pd
from data import ChessDataset, printMove
from torch.utils.data import DataLoader, Dataset

# Data parameters
TRAIN_FILENAMES = ["2023-11", "2023-12"]
EVAL_FILENAMES = ["2023-12"]
WEIGHT_FILEPATH = "weights/best/weights.pth"
ROW_LIMIT = 5000 # Maximum number of games, None for entire file

# Hyperparameters ~35 moves per game 
BATCH_SIZE = 64 # The batch size in moves, 350 kinda worked
SHUFFLE_DATA = True
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001 # 0.0001 kinda worked with batch size 350
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
        x = torch.flatten(x, 1) # Flattens to feed into FC layer
        # print(x.shape, metadata.shape)
        # x = torch.cat((x, metadata), dim=1) # Appends metadata
        # print(x.shape)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x

def train(dataset, job_id):
    # Data initialization
    train_data = dataset
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)

    # Model initialization
    model = Model()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

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
            
        torch.save(model.state_dict(), f'./weights/job_{job_id}/model_weights{epoch}.pth')
    
    print("Finished training at time", timeSinceStart())

def trainMode(job_id):
    print(f'Starting batch job: {job_id}')
    print("===Creating Dataset===")
    train_data = ChessDataset(TRAIN_FILENAMES, ROW_LIMIT)
    print("===Beginning Training===")
    train(train_data, job_id)

def evalMode(job_id):
    print("===Loading Eval Set===")
    # for now, we are only loading 1 games worth, and with a batch size of 1
    eval_data = ChessDataset(TRAIN_FILENAMES, 1)
    eval_loader = DataLoader(eval_data, batch_size=12, shuffle=False)
    print("===Loading Trained Model===")
    loaded_model = Model()
    loaded_model.load_state_dict(torch.load(WEIGHT_FILEPATH))
    loaded_model.eval()
    print("===Making Prediciton===")
    x, metadata, y  = next(iter(eval_loader))
    print(x.shape)
    print(x[11:].shape)
    pred = loaded_model(x[11:], metadata[11:])
    printMove(x[11:])
    print("=== EXPECTED ANSWER ===")
    print(y[11].view(2,8,8))
    print("=== OUTPUT ANSWER ===")
    print(pred.shape)
    print(pred.view(2, 8, 8))

if(__name__ == "__main__"):
    args = sys.argv
    print(f'arguments: {args}')
    # first is name of file, second is job_id, third is mode (true = train)

    if len(args) == 3:
        if args[2] == 'False':
            evalMode(args[1])
        else:
            trainMode(args[1])
    else:
        trainMode(args[1])


    