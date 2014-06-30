#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
NAME=ransuq_$1
srun mainscript -location=cluster -j=$NAME --error=$NAME_error.txt --output=$NAME_output.txt