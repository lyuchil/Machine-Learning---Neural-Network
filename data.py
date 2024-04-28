import csv
import json
import signal
import chess
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

# filenames is a list of filenames
class ChessDataset(Dataset):
    def __init__(self, filenames, row_limit):
        self.row_limit = row_limit
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
            print(f'Loading from {file[19:]}')
            game_df = pd.read_csv(file, nrows=self.row_limit) # dataframe
            for index, game in game_df.iterrows():
                if index % 200 == 0:
                    print(f'{index} games loaded!')
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

        try:
            move = eval("{" + move + "}") # Python magic that converts string to dict
            if(not move["move"]):
                continue
            x = np.array(json.loads(move["tensor"])) # Converts from string to list
            x = unflatten(x, (8, 8, 18))
            y = move_to_tensor(move["move"])
            y = y.reshape(128)
            x_tensor.append(x)
            y_tensor.append(y)
            """
            0th bit -- A1 white queenside 
            7th bit -- H1 white kingside
            56th bit -- A8 black queenside
            63rd bit -- H8 black kingside
            """
            new_castling_rights = [move["castling_right"] & chess.BB_A1,  (move["castling_right"] & chess.BB_H1) >> 6, 
                                   (move["castling_right"] & chess.BB_A8) >> 54, (move["castling_right"] & chess.BB_H8) >> 60 ] 
            metadata = [move["clk"], move["player_to_move"]] + new_castling_rights
            metadata_tensor.append(metadata) 
        except SyntaxError as e:
            print("Syntax error while parsing game", e)

    x_tensor = torch.tensor(np.array(x_tensor)).float()
    y_tensor = torch.tensor(np.array(y_tensor)).float()
    metadata_tensor = torch.tensor(np.array(metadata_tensor)).float()

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

def printMove(move):
    cur_board = np.full((8,8), ' ', dtype='U10')
    prev_board = np.full((8,8), ' ', dtype='U10')
    if len(move.shape) == 4:
        for i in range(0,6):
            for j in range(0,8):
                for k in range(0,8):
                    piece = move[0, i, j, k]
                    prev_piece = move[0, i+6, j, k]
                    if piece == 1:
                        cur_board[j,k] = chess.PIECE_SYMBOLS[i+1].upper()
                    elif piece == -1:
                        cur_board[j,k] = chess.PIECE_SYMBOLS[i+1]
                    if prev_piece == 1:
                        prev_board[j,k] = chess.PIECE_SYMBOLS[i+1].upper()
                    elif prev_piece == -1:
                        prev_board[j,k] = chess.PIECE_SYMBOLS[i+1]

        print("Current Board State")
        print(cur_board)
        print("Previously Moved Pieces")
        print(prev_board)
    else:
        for i in range(move.shape[0]):
            print(f"Layer {i}")
            print(move[i, :, :])
            print('')


    # if len(move.shape) == 4:
    #     for i in range(move.shape[1]):
    #         print(f"Layer {i}")
    #         print(move[0, i, :, :])
    #         print('')