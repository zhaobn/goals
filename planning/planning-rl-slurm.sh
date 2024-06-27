
# ----------- VARIABLES FOR SLURM SCHEDULER -----------
#SBATCH -t 10
#SBATCH --job-name=RL-goal-planner
#SBATCH -o 'slurm-log/rl-planning-%A_%a.out'
#SBATCH -e 'slurm-log/rl-planning-%A_%a.err'
#SBATCH --mail-user=jbbyers@princeton.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=8g

# minutes allocated for job on cluster
#SBATCH --cpus-per-task=4

# Number of subjects to simulate
#SBATCH --array=0-1  #5832
# --------------------------------------------------------------------------------

module load anacondapy/2024.02 # load the system module for conda
conda activate goals-concepts # activate the conda environment for this project
./shapeworld-planning.py $SLURM_ARRAY_TASK_ID

