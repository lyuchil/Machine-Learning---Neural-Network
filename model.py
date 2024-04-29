import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time
import pandas as pd
from data import ChessDataset, printMove
from torch.utils.data import DataLoader, Dataset
"""TODO: 
    Legality checking / reporting
        arg_max (legal) from layer
            from selected piece, arg_max (legal) to_layer
    Implementing accuracy checking during training
    Train a big boi model
"""
# Data parameters
TRAIN_FILENAMES = ["2023-10", "2023-12", "2024-02","2024-03",]
EVAL_FILENAMES = ["2023-11"]
TEST_FILENAMES = ["2024-01"]
WEIGHT_FILEPATH = "weights/job_471557/model_weights9.pth" # best so far "weights/job_471557/model_weights9.pth"
ROW_LIMIT = 1250 # Maximum number of games, None for entire file # 2500 took an hour, 5000 took 2 hours

# Hyperparameters ~35 moves per game 
BATCH_SIZE = 64 # The batch size in moves, 350 kinda worked
SHUFFLE_DATA = True
NUM_EPOCHS = 10
LEARNING_RATE = 5e-6 # 1e-5 best
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
        self.fc1 = nn.Linear(OUT_CHANNELS * 8 * 8 + 6, OUT_CHANNELS * 8 * 8) # 5002, 4096
        self.fc2 = nn.Linear(OUT_CHANNELS * 8 * 8, OUT_CHANNELS * 8 * 8)
        self.fc3 = nn.Linear(OUT_CHANNELS * 8 * 8, 128)

    def forward(self, x, metadata):
        # print(x.shape)
        x = torch.selu(self.conv1(x))
        # print(x.shape)
        x = torch.selu(self.conv2(x))
        # print(x.shape)
        x = torch.selu(self.conv3(x))
        # print(x.shape)
        x = torch.flatten(x, 1) # Flattens to feed into FC layer
        # print(x.shape, metadata.shape)
        x = torch.cat((x, metadata), dim=1) # Appends metadata
        # print(x.shape)

        x = torch.selu(self.fc1(x))
        x = torch.selu(self.fc2(x))
        x = torch.selu(self.fc3(x))
        return x

def train(train_files, eval_files, job_id):
    print("=== TRAINING MODE ===")
    # Data initialization
    print("=== Loading Training and Evaluating Datasets ===")
    train_data = ChessDataset(train_files, ROW_LIMIT)
    eval_data = ChessDataset(eval_files, ROW_LIMIT)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False)

    print("=== Starting Model ===")
    # Model initialization
    model = Model()
    model.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9) #0.9 gave us 1% best

    # Training
    print(f"Started training at {round(timeSinceStart(),2)}")
    for epoch in range(NUM_EPOCHS):
        model.train()
        for x, metadata, y in train_loader:
            x, metadata, y = x.cuda(), metadata.cuda(), y.cuda()
            optimizer.zero_grad()
            predictions = model(x, metadata)
            loss = loss_function(predictions, y)
            loss.backward()
            optimizer.step()
        #Save model state
        torch.save(model.state_dict(), f'./weights/job_{job_id}/model_weights{epoch}.pth')
        # validation after epoch  
        correct, moves = evaluate(model, eval_loader, True)
        print(f'Ended training epoch {epoch+1} at {round(timeSinceStart(),2)} with accuracy: {correct}/{moves} = {round(correct/moves,7)}')
            
    print("Finished training at time", timeSinceStart())
    testMode(TEST_FILENAMES, model, True)

def testMode(test_files, curr_model, cuda_enabled):
    print("=== TEST MODE ===")
    print("=== Loading Test Set ===")
    # for now, we are only loading 1 games worth, and with a batch size of 1
    test_data = ChessDataset(test_files, ROW_LIMIT)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    # if we already have a model, do not load one from memory
    if curr_model:
        model = curr_model
    else:
        print("=== Loading Trained Model ===")
        model = Model()
        if cuda_enabled:
            model.load_state_dict(torch.load(WEIGHT_FILEPATH))
        else:
            model.load_state_dict(torch.load(WEIGHT_FILEPATH, map_location=torch.device('cpu')))
    print("=== Making Predicitons ===")
    correct, moves = evaluate(model, test_loader, cuda_enabled)
    print(f'Final Accuracy: {correct}/{moves} = {round(correct/moves)}')

def evaluate(model, dataset_loader, cuda_enabled):
    move_counter, correct_counter = 0, 0
    model.eval() # disables training mode. 
    with torch.no_grad():
        for x, metadata, y in dataset_loader:
            if cuda_enabled:
                x, metadata, y = x.cuda(), metadata.cuda(), y.cuda()
            pred = model(x, metadata)
            predicted_move = selectMove(pred)
            if cuda_enabled:
                predicted_move = predicted_move.cuda()
            # print("Pred vs predicted_move")
            # print(pred.view(2,8,8))
            # print(predicted_move.view(2,8,8))
            # print("Prediction - Answer")
            # diff = predicted_move - y
            # print(diff.view(2,8,8))
            if torch.equal(predicted_move,y):
                correct_counter +=1
            move_counter +=1
    return correct_counter, move_counter
    
def selectMove(prediction):
    # select largest from both layers
    # set everything to 0 but those two vals
    # Expects a model forward result in the form of a [1,126]

    from_val = torch.argmax(prediction[0,:63])
    to_val = torch.argmax(prediction[0,64:])

    pred_move = torch.zeros((1,128))
    pred_move[0, from_val] = 1
    pred_move[0, to_val + 64] = 1
    return pred_move


if(__name__ == "__main__"):
    args = sys.argv
    # first is name of file, second is job_id, third is mode (true = train)
    print(f'Starting batch job: {args[1]}')
    if len(args) > 2 and args[2] == 'False':
            testMode(TEST_FILENAMES, None, False)
    else:
        train(TRAIN_FILENAMES, EVAL_FILENAMES, args[1])
    