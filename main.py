import chess.pgn
import csv


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

# result list
all_games = []

data = open("./lichess_db_standard_rated_2018-06.pgn", encoding='utf-8')

val = 0

game_count = 0

# the outer while loop for val <= is for testing purpose to limit the amount it runs just to see if it works
# for actual parsing purpose, this should just be a while true loop
while val <= 10:
    # iterating through all the games in the file
    # break when there are no more games
    game = chess.pgn.read_game(data)
    if game is None:
        break

    b_elo = game.headers['BlackElo']
    w_elo = game.headers['WhiteElo']

    average_elo = (int(b_elo) + int(w_elo)) / 2
    
    # this condition can be changed depending the elo range we want to parse and other conditions
    if game.headers['TimeControl'] == "300+0" and game.headers['Termination'] == "Normal" and average_elo >= 1100 and average_elo <= 1200:

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
        prev_move = None # For storing previous moves
    
        # iterate through each node for every possible move
        while node is not None:

            # if the comment is none, likely the start of the game
            if not node.comment:
                clk = '[%clk 0:05:00]' # TODO change to be variable based on time control
            else:
                clk = node.comment

            current_move_data = {
                "clk" : clk,
                "board_fen" : node.board().board_fen(), # Board state
                "player_to_move" : node.turn(),         # True is white, False is black
                "move" : None,                          # Assigned later when looking at the next position
                "opponent_prev_move" : str(node.move),  # The move *leading* to the current position (opponent's move)
                "player_prev_move" : str(prev_move)     # The move before that, (player's last move)
            }

            if(prev_dict):
                prev_dict["move"] = str(node.move)
           
            #temp_dict = create_something(node, board_fen)
            #list.append(temp_dict)

            # appending every move to the list
            game_data['moves'].append(current_move_data)

            # iterate to the next node
            prev_dict = current_move_data
            prev_move = node.move
            node = node.next()

        all_games.append(game_data)
        val += 1   
        game_count += 1
    else:
        continue


# write to csv file using the given file name
csv_path = "output.csv"

fields = all_games[0].keys()

with open(csv_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    
    # Write header
    writer.writeheader()
    
    # Write rows
    for row in all_games:
        writer.writerow(row)

print("CSV file written successfully.")







  
    

