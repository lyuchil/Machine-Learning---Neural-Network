import chess.pgn
import csv

list = []

def create_something(node, board_fen):

    if not node.comment:
        clk = '[%clk 0:05:00]'
    else:
        clk = node.comment


    dict = {
        "clk" : clk,
        "board_fen" : board_fen
    }

    return dict


data = open("C:/Users/Yu-Chi/Downloads/lichess_db_standard_rated_2024-02.pgn", encoding='utf-8')

val = 0

while val <= 10:
    game = chess.pgn.read_game(data)
    if game is None:
        break

    b_elo = game.headers['BlackElo']
    w_elo = game.headers['WhiteElo']

    average_elo = (int(b_elo) + int(w_elo)) / 2
    

    

    if game.headers['TimeControl'] == "300+0" and game.headers['Termination'] == "Normal" and average_elo >= 1100 and average_elo <= 1200:
        node = game
    
        while node is not None:
            board = node.board()
            board_fen = board.board_fen()
        

           
            temp_dict = create_something(node, board_fen)
            list.append(temp_dict)

            node = node.next()

        val += 1   
        
    else:
        continue


csv_path = "output.csv"

fields = list[0].keys()

with open(csv_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    
    # Write header
    writer.writeheader()
    
    # Write rows
    for row in list:
        writer.writerow(row)

print("CSV file written successfully.")







  
    

