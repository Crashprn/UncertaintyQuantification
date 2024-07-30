#!/bin/bash

#PBS -A BuoyDrivenFlows
#PBS -l walltime=24:00:00
#PBS -l select=10:ncpus=32:ngpus=4:system=polaris
#PBS -l filesystems=home:grand
#PBS -q prod
#PBS -o out.txt
#PBS -e err.txt


module use /soft/modulefiles
module load conda
conda activate base
cd ${PBS_O_WORKDIR}
source ./venvs/2024-04-29/bin/activate
sleep 1
python3 -u polaris_hmc_main.py
sleep 1
exit 0
