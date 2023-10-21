#!/bin/bash

#SBATCH --job-name=ass2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:02:00
#SBATCH --partition=cascade

module load Clang/13.0.1

make clean
make solution

for NP in 64
do
	echo $NP cores
	srun -n $NP solution input.txt
done
