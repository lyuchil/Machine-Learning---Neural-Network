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
    Implementing accuracy checking during training
    Giving wieght file to front-end peeps
    Train a big boi model
    Implement Metadata
"""
# Data parameters
TRAIN_FILENAMES = ["2023-11", "2023-12"]
EVAL_FILENAMES = ["2024-01"]
WEIGHT_FILEPATH = "weights/job_470735/model_weights9.pth"
ROW_LIMIT = 2500 # Maximum number of games, None for entire file # 2500 took an hour, 5000 took 2 hours

# Hyperparameters ~35 moves per game 
BATCH_SIZE = 64 # The batch size in moves, 350 kinda worked
SHUFFLE_DATA = True
NUM_EPOCHS = 10
LEARNING_RATE = 0.00001 # 0.0001 kinda worked with batch size 350
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
        self.fc1 = nn.Linear(OUT_CHANNELS * 8 * 8, OUT_CHANNELS * 8 * 8) # 4099, 4096
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
        # x = torch.cat((x, metadata), dim=1) # Appends metadata
        # print(x.shape)

        x = torch.selu(self.fc1(x))
        x = torch.selu(self.fc2(x))
        x = torch.selu(self.fc3(x))
        return x

def train(dataset, job_id):
    # Data initialization
    train_data = dataset
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATA)

    # Model initialization
    model = Model()
    model.cuda()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9) # SGD instead of ADAM

    # Training
    for epoch in range(NUM_EPOCHS):
        print(f"Started epoch {epoch+1} at time {timeSinceStart()}")
        model.train()
        for x, metadata, y in train_loader:
            x, metadata, y = x.cuda(), metadata.cuda(), y.cuda()
            optimizer.zero_grad()
            predictions = model(x, metadata)
            loss = loss_function(predictions, y)
            loss.backward()
            optimizer.step()
            
        torch.save(model.state_dict(), f'./weights/job_{job_id}/model_weights{epoch}.pth')
    
    print("Finished training at time", timeSinceStart())

def trainMode(job_id):
    print(f'Starting batch job: {job_id}')
    print("===Creating Dataset===")
    train_data = ChessDataset(TRAIN_FILENAMES, ROW_LIMIT)
    print("===Beginning Training===")
    train(train_data, job_id)

def evalMode(job_id):
    print("===Loading Eval Set===")
    # for now, we are only loading 1 games worth, and with a batch size of 1
    eval_data = ChessDataset(EVAL_FILENAMES, 100)
    eval_loader = DataLoader(eval_data, batch_size=1, shuffle=False)
    print("===Loading Trained Model===")
    loaded_model = Model()
    loaded_model.load_state_dict(torch.load(WEIGHT_FILEPATH, map_location=torch.device('cpu')))
    loaded_model.eval() # disables training mode. 
    print("===Making Predicitons===")
    # x, metadata, y  = next(iter(eval_loader))
    move_counter = 0
    correct_counter = 0
    for x, metadata, y in eval_loader:
        pred = loaded_model(x, metadata)
        predicted_move = selectMove(pred)
        y_eval = y.view(2,8,8).numpy()
        if np.array_equal(predicted_move,y_eval):
            correct_counter +=1
        move_counter +=1
    
    print(f'Final Accuracy: {correct_counter}/{move_counter} = {correct_counter/move_counter}')
    
def selectMove(prediction):
    # select largest from both layers
    # set everything to 0 but those two vals
    # Expects a model forward result in the form of a [1,126]
    np_pred = prediction.detach().numpy()
    # print(np_pred)
    # print(np_pred.shape)
    # print(np_pred[0,:63])
    # print(np_pred[0,64:])

    from_val = np.argmax(np_pred[0,:63])
    to_val = np.argmax(np_pred[0,64:])
    # print(f'from_val: {from_val} to_val: {to_val}')

    pred_move = np.zeros((2,8,8))
    pred_move[0,from_val//8, from_val % 8] = 1
    pred_move[1,to_val//8, to_val % 8] = 1
    # print(pred_move)
    return pred_move


if(__name__ == "__main__"):
    args = sys.argv
    print(f'arguments: {args}')
    # first is name of file, second is job_id, third is mode (true = train)

    if len(args) == 3:
        if args[2] == 'False':
            evalMode(args[1])
        else:
            trainMode(args[1])
    else:
        trainMode(args[1])


    