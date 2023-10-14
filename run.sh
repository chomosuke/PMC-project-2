#!/usr/bin/env bash
make solution
rm -r points/data
mkdir points/data
mpirun -np 6 solution input.txt
