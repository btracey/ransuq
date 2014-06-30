#!/bin/bash
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
NAME=ransuq_$1
srun ransuqquick -location=cluster -j=$NAME --error=/ADL/btracey/mygo/batch/$NAME_error.txt --output=/ADL/btracey/mygo/batch/$NAME_output.txt