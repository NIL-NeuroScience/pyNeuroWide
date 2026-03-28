#!/bin/bash -l

#$ -N apply_bin_saving
#$ -j y
#$ -pe omp 6
#$ -P devorlab
#% -l buyin

cd /project/devorlab/bcraus/projects/pyNeuroWide
source /project/devorlab/bcraus/envs/pnw/bin/activate

python -u tests/apply_bin_saving.py