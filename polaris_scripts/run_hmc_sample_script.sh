#!/bin/bash

#PBS -A BuoyDrivenFlows
#PBS -l walltime=70:00:00
#PBS -l select=1:ncpus=32:ngpus=4:system=polaris
#PBS -l filesystems=home:grand
#PBS -q preemptable
#PBS -o out_sample.txt
#PBS -e err_sample.txt
#PBS -r y

ndata=15000
chkpt_dir="Model_Checkpoints/HMC_S"

module use /soft/modulefiles
module load conda
conda activate base
cd ${PBS_O_WORKDIR}
source ./venvs/2024-04-29/bin/activate
sleep 1
python3 -u polaris_scripts/polaris_hmc_sample.py --n_data $ndata --chkpt_dir $chkpt_dir
sleep 1
exit 0
