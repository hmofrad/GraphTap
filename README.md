# GraphTap Distributed Graph Analytics System

## Build
    make clean && make

## Run
    mpirun -np 4 bin/./pr   data/rmat10_1024.bin   1024 20
    mpirun -np 4 bin/./sssp data/rmat10_1024_w.bin 1024 0
    mpirun -np 4 bin/./bfs  data/rmat10_1024.bin   1024 0
    mpirun -np 4 bin/./cc   data/rmat10_1024.bin   1024

## Contact
    Mohammad Hasanzadeh Mofrad
    moh18@pitt.edu
