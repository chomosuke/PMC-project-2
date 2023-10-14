#!/usr/bin/env bash
make solution
rm -r points/data
mkdir points/data
mpirun -np 8 solution input.txt
