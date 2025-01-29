#!/bin/sh

# ----------- VARIABLES FOR SLURM SCHEDULER -----------
#SBATCH -t 10
#SBATCH --job-name=RL-goal-planner
#SBATCH --mem=4g
#SBATCH --cpus-per-task=1
##SBATCH --array=0-2181
#SBATCH --array=0-2500

# Comment out logging directives by default
#SBATCH -o 'slurm-log/rl-planning-%A_%a.out'
#SBATCH -e 'slurm-log/rl-planning-%A_%a.err'
#SBATCH --mail-user=jbbyers@princeton.edu
#SBATCH --mail-type=ALL

# Create slurm-log directory if it doesn't exist
mkdir -p slurm-log

# Load conda and activate environment
module load anacondapy/2024.02
if ! conda activate goals-concepts; then
    echo "Failed to activate conda environment"
    exit 1
fi

ACTUAL_TASK_ID=$(($SLURM_ARRAY_TASK_ID + 15001))
python value_iteration.py $ACTUAL_TASK_ID


