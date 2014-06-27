#!/bin/bash
#SBATCH --job-name="ransuq"
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1

echo $1

srun mainscript $1 -location=cluster


#aoseuth SBATCH --output="ransuq_slurm_$NAME.out"
#oeuast SBATCH --error="panic.err"