#!/bin/bash

#PBS -A BuoyDrivenFlows
#PBS -l walltime=24:00:00
#PBS -l select=1:system=polaris
#PBS -l filesystems=home:grand
#PBS -q debug
#PBS -o out.$PBS_JOBID
#PBS -e err.$PBS_JOBID


module use /soft/modulefiles
module load conda
conda activate base
cat ${PBS_O_WORKDIR}
cd ${PBS_O_WORKDIR}
source /venv/2024-04-29/bin/activate
sleep 1
python3 polaris_hmc_main.py
sleep 1
exit 0
EOF

