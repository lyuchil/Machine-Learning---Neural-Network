#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=64g
#SBATCH -J "Parsing Data"
#SBATCH -p academic
#SBATCH -t 24:00:00
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL
#SBATCH --out=slurm_train_chess_model%A.err

cd /home/rpblair/Machine_Learning/Machine-Learning---Neural-Network
source ./runRequirements.sh

# really should pass in stuff properly, oh well

python model.py

