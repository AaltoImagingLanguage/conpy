#!/bin/bash

# Make sure to request only the resources you really need to avoid cueing
#SBATCH -t 5:00
#SBATCH --mem-per-cpu=4G
#SBATCH -n 1

# Location to write the logfile to
LOG_FILE=logs/select_vertices.log

# Load the python environment
module load anaconda3
module load mesa

# Tell BLAS to only use a single thread
export OMP_NUM_THREADS=1

# Start a virtual framebuffer to render 3D things to
Xvfb :99 -screen 0 1400x900x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99.0

# Run the analysis!
srun -o $LOG_FILE python ../08_select_vertices.py
