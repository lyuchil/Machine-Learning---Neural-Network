import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv

FILENAME = "parsed_data_2018-06.csv"

# Hyperparameters
BATCH_SIZE = 100
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
OUT_CHANNELS = 64
KERNAL_SIZE = 3


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(6, OUT_CHANNELS, KERNAL_SIZE)
        self.conv2 = nn.Conv2d(OUT_CHANNELS, OUT_CHANNELS, KERNAL_SIZE) 
        self.conv3 = nn.Conv2d(OUT_CHANNELS, OUT_CHANNELS, KERNAL_SIZE)
        self.fc1 = nn.Linear(OUT_CHANNELS * 8 * 8, 128)
        self.fc2 = nn.Linear(OUT_CHANNELS * 8 * 8, 128)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128) # Flattens to feed into FC layer
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x
    
file = None
csv_reader = None
def load_game():
    global file, csv_reader
    print("Loading game")
    if not file:
        file = open("parsed/" + FILENAME, mode="r")
        print("open")
        csv_reader = csv.reader(file)
        print(next(csv_reader))
    game = next(csv_reader)
    moves = game[2]
    print(moves)

    file.close()

def load_batch():
    file.close()
    
def unflatten(self, flattened_data, shape):
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
        return array_3d

def train():
    model = Model()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()


if(__name__ == "__main__"):
    print(load_game())