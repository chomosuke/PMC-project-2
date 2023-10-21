#!/bin/bash

#SBATCH --job-name=ass2
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:50:00
#SBATCH --partition=cascade

module load Clang/13.0.1

make clean
make solution

for NP in 1
do
	echo $NP cores
	srun -n $NP solution input.txt
done
