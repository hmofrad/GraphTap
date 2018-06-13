#!/bin/bash
# Ubuntu run script

#export PATH=/usr/local/openmpi/bin:$PATH
export PATH=/home/moh/mpich/bin:$PATH:$PATH
#which mpirun
#mpirun -np 4 bin/graph_analytics/pr data/graph_analytics/g1_8_8_13.bin 8 1
mpirun -np 4 bin/graph_analytics/pr ~/graph500/rmat/rmat20.bin 1048576 20
