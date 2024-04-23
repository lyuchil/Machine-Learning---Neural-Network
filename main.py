import chess.pgn
import numpy as np
import sys
import csv

"""
Tensor layers:
0 - current pawns
1 - current knights
2 - current bishops
3 - current rooks
4 - current queen
5 - current king
6 - from pawns
7 - from knights
8 - from bishops
9 - from rooks
10 - from queen
11 - from king
12 - to pawns
13 - to knights
14 - to bishops
15 - to rooks
16 - to queen
17 - to king
"""

class board_tensor:
    def __init__(self, board, previous_move, previous_move_piece, previous2_move, previous2_move_piece):
        self.board = board
        self.prevM = previous_move
        self.prev2M = previous2_move
        self.prevM_piece = previous_move_piece
        self.prev2M_piece = previous2_move_piece
        self.promotionFlag = False
        self.create_tensor()
        
        
    def create_tensor(self):
        
        self.tensor = np.zeros((8,8,18))
        for move in chess.SQUARES:
            piece = self.board.piece_at(move)
            if piece:
                piece_layer = piece.piece_type - 1
                if piece.color:
                    piece_color = 1
                else:
                    piece_color = -1
                row = move // 8
                col = move % 8
                self.tensor[row, col, piece_layer] = piece_color
        
        if self.prev2M:

            from_2 = self.prev2M.from_square
            to_2 = self.prev2M.to_square

            p2_color = 1 if self.prev2M_piece.color else -1

            if self.prev2M.promotion:
                prev2_to_piece_type = self.prev2M.promotion
                self.prev2M_piece.piece_type = 1
            else:
                prev2_to_piece_type = self.prev2M_piece.piece_type
            
            from_row = from_2 // 8
            from_col = from_2 % 8
            self.tensor[from_row, from_col, self.prev2M_piece.piece_type - 1 + 6] = p2_color
            
            to_row = to_2 // 8
            to_col = to_2 % 8
            self.tensor[to_row, to_col, prev2_to_piece_type - 1 + 12] = p2_color
            

        if self.prevM:
            from_p = self.prevM.from_square
            to_p = self.prevM.to_square

            p_color = 1 if self.prevM_piece.color else -1
            
            if self.prevM.promotion:
                prevM_to_piece = self.prevM.promotion
                self.prevM_piece.piece_type = 1
            else:
                prevM_to_piece = self.prevM_piece.piece_type

            from_row = from_p // 8
            from_col = from_p % 8
            self.tensor[from_row, from_col, self.prevM_piece.piece_type - 1 + 6] = p_color
            
            to_row = to_p // 8
            to_col = to_p % 8
            self.tensor[to_row, to_col, prevM_to_piece - 1 + 12] = p_color
    

    def printTensor(self):
        for i in range(self.tensor.shape[2]):
            print(f"Layer {i}")
            print(self.tensor[:, :, i])
            print('')

    def flatten(self):
        # Flatten the 3D array into a 2D list
        flattened_data = [item for sublist in self.tensor
                                 for subsublist in sublist 
                                    for item in subsublist]
        return flattened_data


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
        return np.array(array_3d)

              
all_games = []

data = open(f"./rawGames/lcdb_{sys.argv[1]}-{sys.argv[2]}.pgn", encoding='utf-8')
csv_path = f"./parsed/parsed_data_{sys.argv[1]}-{sys.argv[2]}.csv"

fields = ["game_number", "time_control", "moves"]

with open(csv_path, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    # Write header
    writer.writeheader()
game_count = 0

# the outer while loop for val <= is for testing purpose to limit the amount it runs just to see if it works
# for actual parsing purpose, this should just be a while true loop
while True:
    # iterating through all the games in the file
    # break when there are no more games
    game = chess.pgn.read_game(data)
    if game is None:
        break

    b_elo = game.headers['BlackElo']
    w_elo = game.headers['WhiteElo']

    average_elo = (int(b_elo) + int(w_elo)) / 2

    # 2300 -> 2500 RAPID [600+0]
    # this condition can be changed depending the elo range we want to parse and other conditions
    if game.headers['TimeControl'] == "600+0" and game.headers['Termination'] == "Normal" and average_elo >= 2300 and average_elo <= 2500:

        game_data = {
            "game_number" : game_count,
            "time_control": game.headers['TimeControl'],
            "moves" : []
        }

        # setting the current game as the root node
        node = game

        # The library gives the move *leading* to the current position, rather than the 
        # move *played* in the current position, so we need some extra logic
        prev_dict = None # For storing current moves
        prev_move = None  # For storing previous moves
        prev_move_piece = None
        prev2_move = None # For storing previous 2 moves
        prev2_move_piece = None
        iterator = 0
    
        # iterate through each node for every possible move
        while node is not None:
            # node.move is the move that lead up to this current position AKA the previous move
            #print("node.move " + str(node.move))
            prev2_move = prev_move
            prev2_move_piece = prev_move_piece
            prev_move = node.move
            if node.move:
                prev_move_piece = node.board().piece_at(node.move.to_square)

            tensor = board_tensor(node.board(), prev_move, prev_move_piece, prev2_move, prev2_move_piece)
            # print(node.comment)
            # if the comment is none, likely the start of the game
            # defaulting clk to be max time
            clk = 600 # 600 seconds on the clock default
            if 'clk' in node.comment:
                commSplit = node.comment.split(' ') # gives us ['[%clk', '0:01:26]'] or if eval present: ['[%eval', '0.23]', '[%clk', '0:05:42]']
                time = commSplit[-1].split(']') # gives us ['0:05:42', '']
                hour, min, sec =  time[0].split(':')
                clk = 3600 * int(hour) + 60 * int(min) + int(sec)
                # print(clk)
            # form data here
            
            current_move_data = {
                "clk" : clk,
                "player_to_move" : int(node.turn()),         # True is white, False is black
                "move" : None,                          # Assigned later when looking at the next position
                "tensor" : str(tensor.flatten()),
                "castling_right" : node.board().castling_rights    
            }

            if(prev_dict):
                prev_dict["move"] = str(node.move)
        
            # appending every move to the list
            game_data['moves'].append(current_move_data)

            # iterate to the next node
            prev_dict = current_move_data
            # prev_move = node.move

            node = node.next()

        all_games.append(game_data)
        game_count += 1
        if game_count % 10 == 0:
            print(f'game_count: {game_count}')
            with open(csv_path, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                for row in all_games:
                    writer.writerow(row)

            all_games = []
            
print("CSV file written successfully.")