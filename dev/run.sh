#!/bin/bash
# Ubuntu run script

#export PATH=/home/moh/mpich/bin:$PATH:$PATH
#mpirun -np 4 bin/graph_analytics/degree ~/graph500/rmat/rmat20.bin 1048576 1
mpirun -np 4 bin/graph_analytics/pr data/graph_analytics/g1_8_8_13.bin 8 1
#mpirun -np 4 bin/graph_analytics/pr data/graph_analytics/g1_8_8_13.bin 8 1
