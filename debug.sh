#!/usr/bin/env bash
make debug
mpirun -np 8 valgrind --track-origins=yes ./debug input.txt
