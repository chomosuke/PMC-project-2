#!/bin/bash

#SBATCH --job-name=ass2
#SBATCH --ntasks=64
#SBATCH --nodes=64
#SBATCH --time=00:50:00
#SBATCH --partition=cascade

module load Clang/13.0.1

make clean
make solution

for NP in 2 4 8 16 32 64
do
	echo $NP cores
	srun -n $NP solution input.txt
done
