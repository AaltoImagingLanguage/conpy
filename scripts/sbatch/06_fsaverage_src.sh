#!/bin/bash

# Make sure to request only the resources you really need to avoid cueing
#SBATCH -t 10:00
#SBATCH --mem-per-cpu=8G
#SBATCH -n 1

# Location to write the logfile to
LOG_FILE=logs/fsaverage_src.log

# Load the python environment
module load anaconda3

# Tell BLAS to only use a single thread
export OMP_NUM_THREADS=1

# Run the analysis!
srun -o $LOG_FILE python ../06_fsaverage_src.py
