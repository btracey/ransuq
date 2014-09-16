#!/bin/bash
#SBATCH --job-name="ransuq"
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
echo $1
srun mainscript $1 -location=cluster