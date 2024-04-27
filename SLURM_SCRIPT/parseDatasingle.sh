#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=32g
#SBATCH -J "Parsing Data"
#SBATCH -p academic
#SBATCH -t 24:00:00
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL
#SBATCH --error=slurm_parse_data%A_%a.err
source ./runRequirements.sh

python main.py 2023 12


