#!/bin/bash -l
#SBATCH --job-name=TEACH_P_O_1
# specify number of nodes 
#SBATCH -N 1

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 8

# specify the walltime e.g 20 mins
#SBATCH -t 120:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qiang.wang@ucdconnect.ie

# run from current directory
cd $SLURM_SUBMIT_DIR

module load singularity/3.5.2

# The following commands are informed by:
# https://people.tuebingen.mpg.de/felixwidmaier/rrc2021/singularity.html#build-and-run-code-in-singularity

singularity run /home/people/17203085/rrc_phase1_used/user_image.sif mpirun -np 8 python3 train.py --domain-randomization=0 --increase-fps=1 --difficulty=4 --reward-type='p_o' --teach-collect=1 --teach-epoch=50 --seed=123 --exp-dir='TEACH_P_O_1' --n-epochs=800 2>&1 | tee TEACH_P_O_1.log

