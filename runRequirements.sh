#!/bin/bash
module load python/3.9.12 py-pip py-numpy/1.24.3/
module load zstd py-requests/2.26.0/
module load cuda/11.8.0 # cuda/11.8

pip install -q -r requirements.txt 

# PATH=$PATH:~/.local/lib/python3.9/




