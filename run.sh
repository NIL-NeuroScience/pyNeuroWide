#!/bin/bash -l

#$ -N apply_bin_saving
#$ -j y
#$ -pe omp 4
#$ -P devorlab
#% -l buyin

cd /projectnb/devorlab/bcraus/AnalysisCode/pyNeuroWide
source .venv/bin/activate

python -u tests/apply_bin_saving.py