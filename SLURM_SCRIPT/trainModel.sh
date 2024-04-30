#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=128g
#SBATCH -J "Training Time - 1e-5"
#SBATCH -p academic
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL
#SBATCH --out=slurm_train_chess_model%A.out
#SBATCH --error=slurm_train_chess_model%A.err

cd /home/rpblair/Machine_Learning/Machine-Learning---Neural-Network
source ./runRequirements.sh

# really should pass in stuff properly, oh well
echo "Starting python"
mkdir weights/job_$SLURM_JOB_ID 
python -u model.py $SLURM_JOB_ID True

