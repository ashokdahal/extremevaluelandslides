#!/bin/bash

#SBATCH --ntasks=1          #number of processes to launch
#SBATCH --cpus-per-task=16   #number of threads each task will spawn
#SBATCH --reservation=AI_COMP
#SBATCH --time=10:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
source activate mlashok

srun -n ${SLURM_NTASKS} -c ${SLURM_CPUS_PER_TASK} python run.py
