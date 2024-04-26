import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import csv
import sys
import chess
import json
import time
import signal
import pandas as pd

FILENAME = "parsed_data_2018-06.csv"

# Hyperparameters
BATCH_SIZE = 150 # The batch size in number of **games** (not moves, so the tensor fed into the model is bigger)
SHUFFLE_DATA = False
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
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
        self.fc1 = nn.Linear(OUT_CHANNELS * 8 * 8 + 3, OUT_CHANNELS * 8 * 8)
        self.fc2 = nn.Linear(OUT_CHANNELS * 8 * 8, 128)

    def forward(self, x, metadata):
        print(x.shape)
        x = torch.relu(self.conv1(x))
        print(x.shape)
        x = torch.relu(self.conv2(x))
        print(x.shape)
        x = torch.relu(self.conv3(x))
        print(x.shape)
        x = x.view(x.shape[0], -1) # Flattens to feed into FC layer
        print(x.shape, metadata.shape)
        x = torch.cat((x, metadata), dim=1) # Appends metadata
        print(x.shape)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
    
class ChessDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename
        self.dataframe = pd.read_csv("parsed/" + self.filename)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        game = self.dataframe.iloc[index]["moves"]
        return parse_game(game)

def parse_game(game):
    x_tensor = []
    y_tensor = []
    metadata_tensor = []

    moves = game.split("}, {")

    for move in moves:
        if move[0:2] == "[{":
            move = move[2:]
        elif move[-2:] == "}]":
            move = move[:-2]


        move = eval("{" + move + "}") # Python magic that converts string to dict

        if(not move["move"]):
            continue
        x = np.array(json.loads(move["tensor"])) # Converts from string to list
        x = unflatten(x, (8, 8, 18))
        y = move_to_tensor(move["move"])
        y = y.reshape(128)
        x_tensor.append(x)
        y_tensor.append(y)
        metadata_tensor.append([move["clk"], move["player_to_move"], 0]) # TODO add back castling right

    x_tensor = torch.tensor(x_tensor).float()
    y_tensor = torch.tensor(y_tensor).float()
    metadata_tensor = torch.tensor(metadata_tensor).float()

    return x_tensor, metadata_tensor, y_tensor


def load_batch():
    x_batch = None
    metadata_batch = None
    y_batch = None
    
    file = open("parsed/" + FILENAME, mode="r")
    print("Opened file at time", timeSinceStart())
    csv_reader = csv.reader(file)
    next(csv_reader) # ignore header

    for _ in range(BATCH_SIZE):
        [x, metadata, y] = load_game(csv_reader)
        if(type(x_batch) != np.ndarray):
            x_batch = x
            metadata_batch = metadata
            y_batch = y
        else:
            x_batch = np.concatenate((x_batch, x), axis=0)
            metadata_batch = np.concatenate((metadata_batch, metadata), axis=0)
            y_batch = np.concatenate((y_batch, y), axis=0)

    x_batch = torch.tensor(x_batch).float()
    metadata_batch = torch.tensor(metadata_batch).float()
    y_batch = torch.tensor(y_batch).float()
    
    print("Batch finished loading at time", timeSinceStart())
    return x_batch, metadata_batch, y_batch
        

def load_game(csv_reader):
    x_tensor = []
    y_tensor = []
    metadata_tensor = []

    csv.field_size_limit(sys.maxsize)
    game = next(csv_reader)

    moves = game[2].split("}, {")
    for move in moves:
        if move[0:2] == "[{":
            move = move[2:]
        elif move[-2:] == "}]":
            move = move[:-2]
        move = eval("{" + move + "}")
        if(not move["move"]):
            continue
        x = np.array(json.loads(move["tensor"])) # Converts from string to list
        x = unflatten(x, (8, 8, 18))
        y = move_to_tensor(move["move"])
        y = y.reshape(128)
        x_tensor.append(x)
        y_tensor.append(y)
        metadata_tensor.append([move["clk"], move["player_to_move"], move["castling_right"]])

    x_tensor = np.array(x_tensor)
    y_tensor = np.array(y_tensor)
    metadata_tensor = np.array(metadata_tensor)

    return x_tensor, metadata_tensor, y_tensor
    
# Converts a move string such as "e2e4" into the appropriate tensor
def move_to_tensor(move_str):
    from_square = chess.parse_square(move_str[:2])
    to_square =chess.parse_square(move_str[2:4])

    from_tensor = np.zeros((8, 8))
    to_tensor = np.zeros((8, 8))

    from_tensor[from_square // 8][from_square % 8] = 1
    to_tensor[to_square // 8][to_square % 8] = 1

    return np.array([from_tensor, to_tensor])
    
def unflatten(flattened_data, shape):
        # Reconstruct the 3D array from the flattened data
        array_3d = []
        index = 0
        for i in range(shape[0]):
            subarray_2d = []
            for j in range(shape[1]):
                subsubarray_1d = []
                for k in range(shape[2]):
                    subsubarray_1d.append(flattened_data[index])
                    index += 1
                subarray_2d.append(subsubarray_1d)
            array_3d.append(subarray_2d)

        outp = np.array(array_3d)
        return outp.transpose(2, 0, 1) 

def train():

    # Data initialization
    train_data = ChessDataset(FILENAME)
    train_loader = DataLoader(train_data, batch_size=1, shuffle=SHUFFLE_DATA)

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
            predictions = model(x[0], metadata[0])
            loss = loss_function(predictions, y[0])
            loss.backward()
            optimizer.step()
    
    print("Finished training at time", timeSinceStart())
    [x, metadata, y] = train_data.__getitem__(0)
    print(model(x[:1], metadata[:1]).view(2, 8, 8))


if(__name__ == "__main__"):
    train()