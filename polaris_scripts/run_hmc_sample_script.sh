#!/bin/bash

#PBS -A BuoyDrivenFlows
#PBS -l walltime=01:00:00
#PBS -l select=1:ncpus=32:ngpus=4:system=polaris
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -o out.txt
#PBS -e err.txt


module use /soft/modulefiles
module load conda
conda activate base
cd ${PBS_O_WORKDIR}
source ./venvs/2024-04-29/bin/activate
sleep 1
python3 -u polaris_scripts/polaris_hmc_sample.py
sleep 1
exit 0
