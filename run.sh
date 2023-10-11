#!/usr/bin/env bash
make solution
mpirun -np 6 solution input.txt
