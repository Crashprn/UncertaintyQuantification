#!/bin/bash


find . -name "*batch*" -delete
current_dir=$(pwd)
procs_per_node=64
NN=1
NP=$(($NN * $procs_per_node))
save_dir="data"
n_data=80000
n_restarts=2
grid_dim=700
verbose=1
dim_y=$1
run_name="D_GT_0"


# Set up batch script:
cat > $1.batch << EOF
#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --nodes=$NN
#SBATCH --mem=500000
#SBATCH -o out.$dim_y
#SBATCH -e err.$dim_y
#SBATCH --account=usumae-np
#SBATCH --partition=usumae-np
module load miniconda3/latest
conda activate base
cd $current_dir
sleep 5
python3 -u UT_hpc/chpc_gp_main.py --save_dir $save_dir --n_data $n_data --n_restarts $n_restarts --grid_dim $grid_dim --verbose $verbose --dim_y $dim_y --run_name $run_name
sleep 5
exit 0
EOF

cat $1.batch
echo $current_dir
sbatch $1.batch
sleep 5
