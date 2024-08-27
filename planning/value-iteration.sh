#!/bin/sh

# ----------- VARIABLES FOR SLURM SCHEDULER -----------
#SBATCH -t 10
#SBATCH --job-name=RL-goal-planner
#SBATCH -o 'slurm-log/rl-planning-%A_%a.out'
#SBATCH -e 'slurm-log/rl-planning-%A_%a.err'
#SBATCH --mail-user=jbbyers@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=1g

# minutes allocated for job on cluster
#SBATCH --cpus-per-task=4

# Number of subjects to simulate
#SBATCH --array=0-2500
# --------------------------------------------------------------------------------
#5832 = total states
module load anacondapy/2024.02 # load the system module for conda
conda activate goals-concepts # activate the conda environment for this project
# Calculate the actual task ID
ACTUAL_TASK_ID=$(($SLURM_ARRAY_TASK_ID + 0))
python value_iteration.py $ACTUAL_TASK_ID

