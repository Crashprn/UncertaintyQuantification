#!/bin/bash

find . -name "*batch" -delete
current_dir=$(pwd)
procs_per_node=16
NN=1
NP=$(($NN * $procs_per_node))
save_dir="data/GP_Pytorch/D_LT_0_Alea_No_Est"
n_data=20000
grid_dim=700
batch_size=26000
max_iter=1000
n_inducing=6000
verbose=1
y_dim=$1
run_name="D_LT_0_Alea_No_Est"
resume=0


# Set up batch script:
cat > $1.batch << EOF
#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --nodes=$NN
#SBATCH --mem=60000
#SBATCH -o out.$y_dim
#SBATCH -e err.$y_dim
#SBATCH --account=usumae-kp
#SBATCH --partition=usumae-kp
module load miniconda3/latest
conda activate base
cd $current_dir
sleep 5
python3 -u UT_hpc/chpc_gpytorch_main.py --save_dir $save_dir --n_data $n_data --grid_dim $grid_dim --verbose $verbose --y_dim $y_dim --max_iter $max_iter --run_name $run_name --batch_size $batch_size --n_inducing $n_inducing --resume $resume
sleep 5
exit 0
EOF

cat $1.batch
echo $current_dir
sbatch $1.batch
sleep 5
