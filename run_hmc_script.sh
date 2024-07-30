#!/bin/bash


find . -name "*batch*" -delete
current_dir=$(pwd)
procs_per_node=64
NN=1
NP=$(($NN * $procs_per_node))
warmup_steps=200
samples=1000
num_chains=4
tree_depth=15



# Set up batch script:
cat > $1.batch << EOF
#!/bin/bash
#SBATCH --time=09-00:00:00
#SBATCH --nodes=$NN
#SBATCH --ntasks=$NP
#SBATCH -o out.$NN
#SBATCH -e err.$NN
#SBATCH --account=usumae-np
#SBATCH --partition=usumae-np
#SBATCH -C mil|rom
module load miniconda3/latest
conda activate base
cd $current_dir
sleep 5
python3 chpc_main.py --warmup_steps $warmup_steps --num_samples $samples --num_chains $num_chains --tree_depth $tree_depth
sleep 5
exit 0
EOF

cat $1.batch
echo $current_dir
sbatch $1.batch
sleep 5
