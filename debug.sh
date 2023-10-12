#!/usr/bin/env bash
make debug
mpirun -np 3 valgrind --track-origins=yes ./debug input.txt
