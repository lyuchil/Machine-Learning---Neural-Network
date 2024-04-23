#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --mem=32g
#SBATCH -J "Parsing Data"
#SBATCH -p academic
#SBATCH -t 24:00:00
#SBATCH --mail-user=rpblair@wpi.edu
#SBATCH --mail-type=ALL
#SBATCH --array=0-5
#SBATCH --error=slurm_parse_data%A_%a.err

dataYears=("2023" "2023" "2023" "2024" "2024" "2024")
dataMonths=("10" "11" "12" "01" "02" "03") 

source ./runRequirements.sh

echo $SLURM_ARRAY_TASK_ID
python main.py ${dataYears[$SLURM_ARRAY_TASK_ID]} ${dataMonths[$SLURM_ARRAY_TASK_ID]}


