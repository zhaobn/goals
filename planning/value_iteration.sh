#!/bin/sh 

# ----------- VARIABLES FOR SLURM SCHEDULER -----------
#SBATCH -t 5760 # time in minutes
#SBATCH --job-name=value-iteration
#SBATCH -o 'slurm-log/value-iteration-%A_%a.out'
#SBATCH -e 'slurm-log/value-iteration-%A_%a.err'
#SBATCH --mail-user=jbbyers@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=10g
#SBATCH --cpus-per-task=4

module load anacondapy/2024.02 # load the system module for conda
conda activate goals-concepts # activate the conda environment for this project
python3 shapeworld-planning.py

