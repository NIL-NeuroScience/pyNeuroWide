#!/bin/bash -l

#$ -N motion_correction
#$ -j y
#$ -pe omp 6
#$ -P devorlab
#% -l buyin

cd /project/devorlab/bcraus/projects/pyNeuroWide
# source /project/devorlab/bcraus/envs/s2p/bin/activate
module load miniconda
conda activate pnw

python -u examples/processing_2P.py --path /projectnb/devorlab/bcraus/HRF/2P/26-04-10/Rbp4_132