#!/bin/bash

#SBATCH --job-name=ass2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --partition=cascade

module load Clang/13.0.1

make clean
make solution

for NP in 2 4 8 16
do
	echo $NP cores
	srun -n $NP solution input.txt
done
