#!/bin/bash


find . -name "*batch*" -delete
current_dir=$(pwd)
procs_per_node=64
NN=1
NP=$(($NN * $procs_per_node))
save_dir="data"
n_data=80000
n_restarts=10
grid_dim=700
verbose=1


# Set up batch script:
cat > $1.batch << EOF
#!/bin/bash
#SBATCH --time=00-03:00:00
#SBATCH --nodes=$NN
#SBATCH -o out.$NN
#SBATCH -e err.$NN
#SBATCH --account=dutta
#SBATCH --partition=kingspeak
module load miniconda3/latest
conda init
conda activate base
cd $current_dir
sleep 5
python3 chpc_gp_main.py --save_dir $save_dir --n_data $n_data --n_restarts $n_restarts --grid_dim $grid_dim --verbose $verbose
sleep 5
exit 0
EOF

cat $1.batch
echo $current_dir
sbatch $1.batch
sleep 5
