#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=32g
#SBATCH -J "Downloading Data"
#SBATCH -p academic
#SBATCH -t 24:00:00
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL

./runRequirements.sh

./runDownload.sh 2018 06


