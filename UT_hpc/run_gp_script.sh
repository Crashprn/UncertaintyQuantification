#!/bin/bash


find . -name "*batch" -delete
current_dir=$(pwd)
procs_per_node=16
NN=1
NP=$(($NN * $procs_per_node))
save_dir="data/Aleatoric/Regular/"
n_data=20000
grid_dim=700
batch_size=15000
max_iter=1500
verbose=1
dim_y=$1
run_name="Regular_Alea"
resume=0


# Set up batch script:
cat > $1.batch << EOF
#!/bin/bash
#SBATCH --time=4-00:00:00
#SBATCH --nodes=$NN
#SBATCH --mem=60000
#SBATCH -o out.$dim_y
#SBATCH -e err.$dim_y
#SBATCH --account=usumae-kp
#SBATCH --partition=usumae-kp
module load miniconda3/latest
conda activate base
cd $current_dir
sleep 5
python3 -u UT_hpc/chpc_gp_main.py --save_dir $save_dir --n_data $n_data --grid_dim $grid_dim --verbose $verbose --dim_y $dim_y --max_iter $max_iter --run_name $run_name --batch_size $batch_size --resume $resume
sleep 5
exit 0
EOF

cat $1.batch
echo $current_dir
sbatch $1.batch
sleep 5
