import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import time
import pandas as pd
from data import ChessDataset, printMove, selectMove, find_legal_move
from torch.utils.data import DataLoader, Dataset

# Data parameters
TRAIN_FILENAMES = ["2023-10", "2023-12", "2024-02","2024-03"]
EVAL_FILENAMES = ["2023-11"]
TEST_FILENAMES = ["2024-01"]
WEIGHT_FILEPATH = "weights/job_473513/model_weights12.pth" 
ROW_LIMIT = 5000 
DEBUG_FLAG = False

# Hyperparameters ~35 moves per game 
BATCH_SIZE = 64 # The batch size in moves, 350 kinda worked
SHUFFLE_DATA = True
NUM_EPOCHS = 50
LEARNING_RATE = 1e-2 # best resutls were somwhere in between 1-e1 and 1e-2
MOMENTUM = 0.9
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

        self.fc3_1_1 = nn.Linear(OUT_CHANNELS * 8 * 8, OUT_CHANNELS * 8 * 8)
        self.fc3_1_2 = nn.Linear(OUT_CHANNELS * 8 * 8, 64)

        self.fc3_2_1 = nn.Linear(OUT_CHANNELS * 8 * 8, OUT_CHANNELS * 8 * 8)
        self.fc3_2_2 = nn.Linear(OUT_CHANNELS * 8 * 8, 64) 

    def forward(self, x, metadata):
        # conv layers
        x = torch.selu(self.conv1(x))
        x = torch.selu(self.conv2(x))
        x = torch.selu(self.conv3(x))

        # concat layer for metadata
        x = torch.flatten(x, 1) # Flattens to feed into FC layer
        x = torch.cat((x, metadata), dim=1) # Appends metadata
        
        # fully connected linear layers
        x = torch.selu(self.fc1(x))
        x = torch.selu(self.fc2(x))

        #split fully connected linear layers
        x_1 = torch.selu(self.fc3_1_1(x))
        x_1 = self.fc3_1_2(x_1)

        x_2 = torch.selu(self.fc3_2_1(x))
        x_2 = self.fc3_2_2(x_2)

        return x_1, x_2

def train(train_files, eval_files, job_id):
    print("=== TRAINING MODE ===")
    # Data initialization
    print("=== Loading Training and Evaluating Datasets ===")
    train_data = ChessDataset(train_files, ROW_LIMIT)
    eval_data = ChessDataset(eval_files, ROW_LIMIT/4)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False)

    print("=== Starting Model ===")
    # Model initialization
    model = Model()
    model.cuda()
    from_loss_function = nn.BCEWithLogitsLoss()
    to_loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM) #0.9 gave us 1% best

    # Training
    print(f"Started training at {round(timeSinceStart(),2)}")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        for x, metadata, y_from, y_to in train_loader:
            x, metadata, y_from, y_to = x.cuda(), metadata.cuda(), y_from.cuda(), y_to.cuda()
            optimizer.zero_grad()
            from_pred, to_pred = model(x, metadata)
            from_loss = from_loss_function(from_pred, y_from)
            to_loss = to_loss_function(to_pred, y_to)
            sum_loss = from_loss + to_loss 
            sum_loss.backward()
            epoch_loss += sum_loss.item()
            optimizer.step()
        #Save model state
        torch.save(model.state_dict(), f'./weights/job_{job_id}/model_weights{epoch}.pth')
        # validation after epoch - currently this section encompasses the largest amount of time for training. 
        # Future steps include optimization of this method
        if epoch % 5 == 0: 
            correct, moves = evaluate(model, eval_loader, True)
            print(f'[Epoch {epoch+1}, Time {round(timeSinceStart(),2)}] acc: {correct}/{moves} ({round(correct/moves,5)}) and loss {round(epoch_loss,2)}')
            
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
    print(f'Final Accuracy: {correct}/{moves} = {round(correct/moves, 5)}')

def evaluate(model, dataset_loader, cuda_enabled):
    move_counter, correct_counter = 0, 0
    model.eval() # disables training mode. 
    with torch.no_grad():
        for x, metadata, y_from, y_to in dataset_loader:
            if cuda_enabled:
                x, metadata, y_from, y_to = x.cuda(), metadata.cuda(), y_from.cuda(), y_to.cuda()
            pred_from, pred_to = model(x, metadata)
            actual_from_pred, actual_to_pred, __ = find_legal_move(x, metadata, pred_from, pred_to)
            if cuda_enabled:
                actual_from_pred, actual_to_pred =  actual_from_pred.cuda(), actual_to_pred.cuda()
            if DEBUG_FLAG:
                print(f'afp shape {actual_from_pred.shape}, yearly shape {y_from.shape}, afp shape {actual_to_pred.shape}, yearly shape {y_to.shape}')
                print(f'pred_from')
                print(pred_from.view(1,8,8))
                print(f'pred_to')
                print(pred_to.view(1,8,8))
                print(f'afp')
                print(actual_from_pred.view(1,8,8))
                print(f'from_answer')
                print(y_from.view(1,8,8))
                print(f'actual_to')
                print(actual_to_pred.view(1,8,8))
                print(f'to_answer')
                print(y_to.view(1,8,8))
                exit()

            if torch.equal(actual_from_pred,y_from) and torch.equal(actual_to_pred,y_to):
                correct_counter +=1
            move_counter +=1
    return correct_counter, move_counter

if(__name__ == "__main__"):
    args = sys.argv
    # first is name of file, second is job_id, third is mode (true = train)
    print(f'Starting batch job: {args[1]}')
    if len(args) > 2 and args[2] == 'False':
            testMode(TEST_FILENAMES, None, False)
    else:
        train(TRAIN_FILENAMES, EVAL_FILENAMES, args[1])
    