#!/bin/bash

# Make sure to request only the resources you really need to avoid cueing
#SBATCH -t 15:00
#SBATCH --mem-per-cpu=16G
#SBATCH -n 1

# Do the analysis for each subject. This should correspond with the SUBJECTS
# variable below.
#SBATCH --array=1-16

# The input files for all the subjects.
SUBJECTS=( 
	sub002
	sub003
	sub004
	sub006
	sub007
	sub008
	sub009
	sub010
	sub011
	sub012
	sub013
	sub014
	sub015
	sub017
	sub018	     
	sub019	     
)

# Find the current subject
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID - 1]}

# Location to write the logfile to
LOG_FILE=logs/$SUBJECT-ica.log

# Load the python environment
module load anaconda
module load mesa

# Tell BLAS to only use a single thread
export OMP_NUM_THREADS=1

# Start a virtual framebuffer to render 3D things to
Xvfb :99 -screen 0 1400x900x24 -ac +extension GLX +render -noreset &
export DISPLAY=:99.0

# Run the analysis!
srun -o $LOG_FILE python ../03_ica.py $SUBJECT
