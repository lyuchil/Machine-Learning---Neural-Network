import csv
import json
import signal
import chess
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

FILENAMES = ["2023-10", "2023-11"]

# filenames is a list of filenames
class ChessDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = []
        for fn in filenames:
            self.filenames.append("parsed/parsed_data_" + fn + ".csv")
        self.data_list = self.load_game_data()

    # TODO: add flavor for timing progress etc
    # tentative add multithreading for parsing...?
    def load_game_data(self):
        print("===LOADING GAMES!===")
        x_list = []
        metadata_list = []
        y_list = []
        # take every game, extract moves into a GIANT list, 
        for file in self.filenames:
            game_df = pd.read_csv(file, nrows=60) # dataframe
            for index, game in game_df.iterrows():
                [x, metadata, y] = parse_game(game)
                x_list.append(x)
                metadata_list.append(metadata)
                y_list.append(y)
        x_list = torch.cat(x_list, dim=0)
        metadata_list = torch.cat(metadata_list, dim=0)
        y_list = torch.cat(y_list, dim=0)
        return [x_list, metadata_list, y_list]

    def __len__(self):
        return len(self.data_list[0])

    def __getitem__(self, index):
        return self.data_list[0][index], self.data_list[1][index], self.data_list[2][index]

def parse_game(game):

    game = game["moves"]
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

    x_tensor = torch.tensor(np.array(x_tensor)).float()
    y_tensor = torch.tensor(np.array(y_tensor)).float()
    metadata_tensor = torch.tensor(np.array(metadata_tensor)).float()

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