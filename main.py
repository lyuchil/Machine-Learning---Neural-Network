import chess.pgn
import csv

# result list
list = []

# def create_something(node, board_fen):

#     if not node.comment:
#         clk = '[%clk 0:05:00]'
#     else:
#         clk = node.comment


#     dict = {
#         "clk" : clk,
#         "board_fen" : board_fen
#     }

#     return dict


data = open("C:/Users/Yu-Chi/Downloads/lichess_db_standard_rated_2024-02.pgn", encoding='utf-8')

val = 0

game_count = 0

# the outer while loop for val <= is for testing purpose to limit the amount it runs just to see if it works
# for actual parsing purpose, this should just be a while true loop
while val <= 10:
    # iterating throug all the games in the file
    # break when there are no more games
    game = chess.pgn.read_game(data)
    if game is None:
        break

    b_elo = game.headers['BlackElo']
    w_elo = game.headers['WhiteElo']

    average_elo = (int(b_elo) + int(w_elo)) / 2
    
    # this condition can be changed depending the elo range we want to parse and other conditions
    if game.headers['TimeControl'] == "300+0" and game.headers['Termination'] == "Normal" and average_elo >= 1100 and average_elo <= 1200:


        information = {
            "Game#" : game_count,
            "moves" : []
        }

        # setting the current game as the root node
        node = game
    
        # iterate through each node for every possible move
        while node is not None:
            board = node.board()
            board_fen = board.board_fen()

            # if the comment is none, likely the start of the game
            if not node.comment:
                clk = '[%clk 0:05:00]'
            else:
                clk = node.comment


            temp_dict = {
                "clk" : clk,
                "board_fen" : board_fen
            }
           
            #temp_dict = create_something(node, board_fen)
            #list.append(temp_dict)

            # appending every move to the list
            information['moves'].append(temp_dict)

            # iterate to the next node
            node = node.next()

        list.append(information)
        val += 1   
        game_count += 1
    else:
        continue


# write to csv file using the given file name
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







  
    

