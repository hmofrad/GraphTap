#!/bin/bash
# Ubuntu run script

#xport PATH=/usr/local/openmpi/bin:$PATH
export PATH=/home/mohammad/mpich/mpich-install/bin:$PATH
mpirun -np 4 bin/graph_analytics/pr data/graph_analytics/g1_8_8_13.bin 8 1
