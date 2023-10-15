#!/bin/bash

#SBATCH --job-name=ass2
#SBATCH --ntasks=16
#SBATCH --nodes=16
#SBATCH --time=00:00:30
#SBATCH --partition=cascade

module load Clang/13.0.1

make clean
make solution

for NP in 2 4 8 16
do
	echo $NP cores
	mpirun -np $NP solution input.txt
done
